# 🎭 random-persona v2 — 实施方案

> 从扁平 4 维随机游走 → 6 层认知-情感-社交架构  
> 实施文档 · 2026-06-18

---

## 目录

1. [总体策略](#1-总体策略)
2. [Phase 1 实施细节](#2-phase-1-实施细节)
3. [Phase 2 实施细节](#3-phase-2-实施细节)
4. [Phase 3 实施细节](#4-phase-3-实施细节)
5. [Phase 4 实施细节](#5-phase-4-实施细节)
6. [测试策略](#6-测试策略)
7. [迁移与兼容](#7-迁移与兼容)
8. [风险与应对](#8-风险与应对)

---

## 1. 总体策略

### 1.1 实施原则

- **每个 Phase 独立可运行**：Phase 1 做完就能加载到 AstrBot 使用，Phase 2-4 是增量升级
- **不引入外部 LLM 调用**：Appraisal、Speech Act、Language Profile 全部规则引擎
- **向后不兼容时做 migration**：`states.json` 格式变化时自动迁移，不丢老数据
- **破坏性最小的改动优先**：Phase 1 重写核心，Phase 2-4 主要是加文件

### 1.2 文件演变

```
v1.0 (当前)                        v2.0 (Phase 1-4 全部完成)
───────────────────────────────    ─────────────────────────────
main.py          275 行            main.py          ~350 行 (hook + commands)
state.py         170 行            state.py         ~400 行 (三层模型)
prompt.py        144 行            prompt.py        ~200 行 (行为指令生成)
silence.py        82 行            [移除，合并到 speech_act]
_conf_schema.json   9 行          _conf_schema.json  ~30 行
metadata.yaml       6 行          metadata.yaml       6 行
__init__.py          1 行         __init__.py          1 行
README.md          90 行          README.md          ~120 行
                                  appraisal.py      ~300 行 (新增)
                                  speech_act.py     ~250 行 (新增)
                                  language.py       ~200 行 (新增)
                                  relationship.py   ~200 行 (新增)
───────────────────────────────    ─────────────────────────────
总计:             777 行          总计:            ~1,950 行
```

### 1.3 分支策略

```
main ─── v1.0.0 ─── phase1 ─── phase2 ─── phase3 ─── phase4 ─── v2.0.0
                       │           │           │           │
                    可加载       可加载      可加载      可加载
```

每个 Phase 合并到 main 后打 tag，Phase 之间不长期维护独立分支。

---

## 2. Phase 1 实施细节

### 2.1 目标

重写情绪状态模型：`单层 dict` → `Trait + Mood + Emotion 三层`。

### 2.2 数据结构

#### states.json v2 格式

```json
{
  "sessions": {
    "<session_id>": {
      "trait": {
        "openness": 0.55, "conscientiousness": 0.60,
        "extraversion": 0.45, "agreeableness": 0.65,
        "neuroticism": 0.40,
        "regulation_style": {
          "reappraisal": 0.55, "suppression": 0.30, "rumination": 0.25
        }
      },
      "mood": {
        "valence": 0.55, "arousal": 0.38, "dominance": 0.50,
        "updated_at": 1718700000
      },
      "emotion": null,
      "meta": {
        "enabled": true,
        "msg_count": 15,
        "last_active": 1718700000
      }
    }
  },
  "relationships": {}
}
```

#### 内存表示（Python dataclass 或 typed dict）

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class EmotionLabel(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

@dataclass
class Trait:
    openness: float = 0.55
    conscientiousness: float = 0.60
    extraversion: float = 0.45
    agreeableness: float = 0.65
    neuroticism: float = 0.40
    # 推导值 (构造函数中计算)
    reappraisal: float = 0.55
    suppression: float = 0.30
    rumination: float = 0.25

    def __post_init__(self):
        # 从 OCEAN 推导调节风格
        self.reappraisal = round(0.3 + self.openness * 0.4 + self.agreeableness * 0.3, 2)
        self.suppression = round(0.1 + self.neuroticism * 0.5 + (1 - self.extraversion) * 0.2, 2)
        self.rumination = round(0.1 + self.neuroticism * 0.6, 2)

    @property
    def mood_baseline(self) -> dict:
        return {
            "valence":     round(0.3 + (1 - self.neuroticism) * 0.4 + self.extraversion * 0.2, 2),
            "arousal":     round(0.3 + self.extraversion * 0.5, 2),
            "dominance":   round(0.3 + self.conscientiousness * 0.3 + (1 - self.neuroticism) * 0.3, 2),
        }

@dataclass
class Mood:
    valence: float = 0.50
    arousal: float = 0.50
    dominance: float = 0.50
    updated_at: float = 0.0

@dataclass
class Emotion:
    primary: Optional[EmotionLabel] = None
    secondary: Optional[EmotionLabel] = None
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    trigger: str = ""
    intensity: float = 0.0
    started_at: float = 0.0
    half_life: float = 300.0

@dataclass
class SessionState:
    trait: Trait
    mood: Mood
    emotion: Optional[Emotion] = None
    meta: dict = field(default_factory=dict)
```

### 2.3 核心算法

#### Mood 的 Ornstein-Uhlenbeck 漂移

```python
import random, math, time

class StateManager:
    # OU 参数
    THETA = 0.15     # 回归强度 (1/小时)
    SIGMA = 0.03     # 波动率 (每小时)

    def drift_mood(self, session_id: str) -> None:
        state = self.get_or_init(session_id)
        mood = state.mood
        baseline = state.trait.mood_baseline

        now = time.time()
        dt_hours = (now - mood.updated_at) / 3600
        if dt_hours <= 0:
            return

        # 离散化 OU 过程
        for dim in ["valence", "arousal", "dominance"]:
            theta = self.THETA
            mu = baseline[dim]
            sigma = self.SIGMA
            x = mood.__dict__[dim]

            # x_{t+dt} = x_t + θ(μ - x_t)dt + σ√dt·ε, ε ~ N(0,1)
            drift = theta * (mu - x) * dt_hours
            noise = sigma * math.sqrt(dt_hours) * random.gauss(0, 1)
            mood.__dict__[dim] = round(max(0.0, min(1.0, x + drift + noise)), 2)

        mood.updated_at = now
```

#### Emotion 的指数衰减

```python
    def decay_emotion(self, session_id: str) -> None:
        state = self.get_or_init(session_id)
        em = state.emotion
        if em is None:
            return

        now = time.time()
        elapsed = now - em.started_at
        intensity = em.intensity * (2 ** (-elapsed / em.half_life))

        if intensity < 0.05:
            # 衰减完毕 → 情绪归入心境
            self._absorb_into_mood(state.mood, em, elapsed)
            state.emotion = None
        else:
            em.intensity = round(intensity, 3)

    def _absorb_into_mood(self, mood: Mood, em: Emotion, duration: float):
        """情绪体验的累积效应反馈到心境"""
        impact = em.intensity * min(1.0, duration / em.half_life) * 0.15
        mood.valence   = round(max(0.0, min(1.0, mood.valence + em.valence * impact)), 2)
        mood.arousal   = round(max(0.0, min(1.0, mood.arousal + em.arousal * impact)), 2)
        mood.dominance = round(max(0.0, min(1.0, mood.dominance + em.dominance * impact)), 2)
```

#### 计算属性

```python
    def patience(self, state: SessionState) -> float:
        t = state.trait
        m = state.mood
        n = state.meta.get("msg_count", 0)

        base = 1.0 - t.neuroticism * 0.5
        mood_penalty = max(0, 0.5 - m.valence) * 0.4
        fatigue = min(0.4, n * 0.015)
        return round(max(0.05, base - mood_penalty - fatigue), 2)

    def silence_threshold(self, state: SessionState) -> float:
        """返回 0.0-1.0，值越高越容易沉默"""
        # baseline 来自 introversion
        base = (1 - state.trait.extraversion) * 0.5
        # low mood → higher threshold
        mood_factor = max(0, 0.5 - state.mood.valence) * 0.4
        # emotion override
        emo_factor = 0
        if state.emotion:
            if state.emotion.primary == EmotionLabel.SADNESS:
                emo_factor = 0.3
            elif state.emotion.primary == EmotionLabel.ANGER:
                emo_factor = 0.2  # might go quiet when angry
        return round(min(1.0, base + mood_factor + emo_factor), 2)
```

### 2.4 main.py 改动

#### 初始化

```python
def __init__(self, context, config=None):
    ...
    self._state_mgr = StateManager(self.data_dir)
    self._prompt = PromptBuilder()
    # silence 逻辑内聚到 state_mgr + prompt，不再独立模块
```

#### on_llm_request hook

```python
@filter.on_llm_request(priority=-100)
async def on_llm_request(self, event, req):
    sid = event.unified_msg_origin
    state = self._state_mgr.get_or_init(sid)

    # 1. 心境漂移
    self._state_mgr.drift_mood(sid)

    # 2. 情绪衰减
    self._state_mgr.decay_emotion(sid)

    # 3. 构建 prompt
    block = self._prompt.build(state)
    if block:
        req.system_prompt += f"\n\n{block}"

    # 4. 交互计数
    state.meta["msg_count"] = state.meta.get("msg_count", 0) + 1
    state.meta["last_active"] = time.time()
```

#### on_llm_response hook

```python
@filter.on_llm_response()
async def on_llm_response(self, event, response):
    sid = event.unified_msg_origin
    state = self._state_mgr.get_state(sid)
    if state is None:
        return

    # 沉默权判断
    silence_chance = self._state_mgr.silence_threshold(state)
    user_msg = getattr(event, "message_str", "") or ""
    if not _has_question(user_msg) and random.random() < silence_chance:
        # 替换为沉默回应
        response.result_chain = self._make_silence_chain(state)
        return

    # 语气词注入（15%概率）
    resp_len = self._maybe_inject_filler(response, state)

    # 状态更新
    state.meta["last_response_len"] = resp_len
    self._state_mgr._maybe_save()
```

### 2.5 命令接口变化

```python
# v1 命令 → v2 命令
/persona              → /persona              (输出改为三层状态)
/persona random       → /persona random       (随机 Trait + Mood)
/persona chill        → /persona chill        (Trait: 低 E + 高 N)
/persona warm         → /persona warm         (Trait: 高 E + 高 A)
/persona talkative    → /persona talkative    (Trait: 高 E + 高 O)
/persona quiet        → /persona quiet        (Trait: 低 E + 低 O)
/persona mode <m>     → [移除]               (被 Speech Act 替代)
/persona off          → /persona off
/persona on           → /persona on
                      → /persona trait <维度> <值>   (新增)
                      → /persona emotion <标签>       (新增: 手动触发情绪)
                      → /persona reset               (新增: 重置关系+状态)
```

### 2.6 数据迁移

```python
def _migrate_from_v1(self):
    """将 v1 的 states.json 迁移到 v2 格式"""
    old_path = os.path.join(self.data_dir, "states.json")
    if not os.path.exists(old_path):
        return

    with open(old_path) as f:
        old_states = json.load(f)

    # 检查是否已是 v2 格式
    first = next(iter(old_states.values()), None)
    if first and "trait" in first:
        return  # already v2

    # 迁移
    for sid, old in old_states.items():
        trait = Trait(
            openness=old.get("openness", 0.55),
            extraversion=old.get("energy", 0.60),      # v1 energy → v2 extraversion
            neuroticism=1 - old.get("patience", 0.72),  # v1 patience 逆映射
            agreeableness=old.get("valence", 0.40) + 0.2,
        )
        trait.__post_init__()

        mood = Mood(
            valence=old.get("valence", 0.40),
            arousal=old.get("energy", 0.60),
            dominance=0.50)
        mood.updated_at = old.get("last_active", time.time())

        self.states[sid] = SessionState(
            trait=trait,
            mood=mood,
            meta={
                "enabled": old.get("enabled", True),
                "msg_count": old.get("msg_count", 0),
                "last_response_len": old.get("last_response_len", 0),
                "last_active": mood.updated_at,
            })

    self.save()

    # 备份旧文件
    os.rename(old_path, old_path + ".v1.bak")
```

### 2.7 验收标准

- [ ] `/persona` 输出显示 Trait/Mood/Emotion 三层状态
- [ ] 心境在多次交互后向 Trait 基线回归（写测试脚本验证 OU 过程）
- [ ] `states.json` 以 v2 格式持久化
- [ ] v1 `states.json` 自动迁移，原有状态映射到 v2
- [ ] `/persona trait extraversion 0.8` 可手动调节特质
- [ ] 新会话的 Trait 随机初始化
- [ ] Phase 1 完成后能在 AstrBot 加载并稳定运行

---

## 3. Phase 2 实施细节

### 3.1 目标

从用户消息中检测 Appraisal 线索，触发离散情绪，替代盲目漂移。

### 3.2 新增文件: `appraisal.py`

#### 检测规则引擎设计

```python
# 规则表: (正则模式, Appraisal 推断, 触发情绪, 强度)
APPRAISAL_RULES = [
    # ── 愤怒 ──
    (r"烦死了|气死|无语|受不了|忍不了",
     {"goal_congruence": -0.7, "agency": "other"},
     EmotionLabel.ANGER, 0.7),

    (r"凭什么|怎么这样|太过分",
     {"goal_congruence": -0.6, "agency": "other", "norm_compatibility": -0.5},
     EmotionLabel.ANGER, 0.6),

    (r"别烦我|滚|闭嘴|走开|够了",
     {"goal_congruence": -0.8, "agency": "other"},
     EmotionLabel.ANGER, 0.8),

    # ── 喜悦 ──
    (r"哈哈哈|笑死|太好笑|233|www",
     {"goal_congruence": 0.6, "expectedness": 0.3},
     EmotionLabel.JOY, 0.5),

    (r"开心|高兴|快乐|幸福|满足|爽",
     {"goal_congruence": 0.7},
     EmotionLabel.JOY, 0.6),

    # ── 悲伤 ──
    (r"难过|伤心|想哭|好累|崩溃|绝望",
     {"goal_congruence": -0.6, "coping_potential": -0.4},
     EmotionLabel.SADNESS, 0.6),

    (r"对不起|我的错|怪我|都是我的问题",
     {"goal_congruence": -0.6, "agency": "self"},
     EmotionLabel.SADNESS, 0.5),

    # ── 恐惧 ──
    (r"怎么办|害怕|好怕|不敢|吓死|担心",
     {"coping_potential": -0.5, "goal_relevance": 0.8},
     EmotionLabel.FEAR, 0.5),

    # ── 惊讶 ──
    (r"卧槽|真的假的|不是吧|天哪|居然|没想到",
     {"expectedness": -0.7},
     EmotionLabel.SURPRISE, 0.6),

    # ── 厌恶 ──
    (r"恶心|恶臭|下头|辣眼睛|吐了",
     {"goal_congruence": -0.5, "norm_compatibility": -0.7},
     EmotionLabel.DISGUST, 0.6),

    # ── 信任/感激 ──
    (r"谢谢你|多亏你|还好有你|相信你|靠谱",
     {"agency": "other", "goal_congruence": 0.8},
     EmotionLabel.TRUST, 0.6),
]
```

#### 情绪强度映射到 PAD

```python
EMOTION_PAD = {
    EmotionLabel.JOY:           (0.70, 0.55, 0.65),
    EmotionLabel.SADNESS:       (0.10, 0.15, 0.20),
    EmotionLabel.ANGER:         (0.10, 0.80, 0.70),
    EmotionLabel.FEAR:          (0.10, 0.75, 0.10),
    EmotionLabel.SURPRISE:      (0.50, 0.70, 0.40),
    EmotionLabel.DISGUST:       (0.10, 0.45, 0.50),
    EmotionLabel.TRUST:         (0.65, 0.35, 0.55),
    EmotionLabel.ANTICIPATION:  (0.55, 0.45, 0.55),
}
```

#### 情绪调节策略选择

```python
REGULATION_STRATEGIES = {
    EmotionLabel.ANGER: [
        ("suppression",     0.3),   # 压抑不表达
        ("reappraisal",     0.4),   # 换角度理解
        ("controlled_expr", 0.3),   # 克制表达
    ],
    EmotionLabel.SADNESS: [
        ("acceptance",      0.4),
        ("distraction",     0.3),
        ("rumination",      0.3),
    ],
    EmotionLabel.JOY: [
        ("amplify",         0.4),
        ("share",           0.5),
        ("moderate",        0.1),
    ],
    EmotionLabel.FEAR: [
        ("seek_reassurance", 0.4),
        ("avoidance",        0.3),
        ("reappraisal",      0.3),
    ],
}

def select_regulation(trait: Trait, emotion_label: EmotionLabel) -> str:
    strategies = REGULATION_STRATEGIES.get(emotion_label, [("acceptance", 1.0)])
    names, base_weights = zip(*strategies)
    weights = list(base_weights)

    # Trait 调节
    if trait.reappraisal > 0.6 and "reappraisal" in names:
        idx = names.index("reappraisal")
        weights[idx] *= 1.5
    if trait.suppression > 0.5 and "suppression" in names:
        idx = names.index("suppression")
        weights[idx] *= 1.5

    return random.choices(names, weights=weights, k=1)[0]
```

### 3.3 main.py hook 改动

```python
@filter.on_llm_request(priority=-100)
async def on_llm_request(self, event, req):
    sid = event.unified_msg_origin
    state = self._state_mgr.get_or_init(sid)
    user_msg = str(getattr(event, "message_str", "") or "")

    # 1. 心境漂移
    self._state_mgr.drift_mood(sid)

    # 2. Appraisal 检测 → 情绪触发 (Phase 2 新增)
    result = self._appraiser.evaluate(user_msg)
    if result and result["trigger"]:
        self._state_mgr.trigger_emotion(sid, result)

    # 3. 情绪衰减
    self._state_mgr.decay_emotion(sid)

    # 4. 构建 prompt（含情绪 + 调节策略）
    regulation = None
    if state.emotion:
        regulation = select_regulation(state.trait, state.emotion.primary)
    block = self._prompt.build(state, regulation)
    if block:
        req.system_prompt += f"\n\n{block}"

    # 5. 计数
    state.meta["msg_count"] = state.meta.get("msg_count", 0) + 1
    state.meta["last_active"] = time.time()
```

### 3.4 验收标准

- [ ] 用户发"烦死了"→ anger 触发 → prompt 反映克制的不满
- [ ] 用户发"哈哈哈"→ joy 触发 → prompt 反映活跃
- [ ] 情绪衰减曲线正确：anger 10 分钟后基本消失
- [ ] 规则引擎覆盖率 ≥ 60%（用一组测试消息验证）
- [ ] 对于无匹配规则的消息，情绪保持 None
- [ ] 多次同向情绪累积 → 心境偏移

---

## 4. Phase 3 实施细节

### 4.1 目标

引入 Speech Act 层和量化语言特征映射，替代 v1 模糊的「五种表达方式」和「回复风格指引」。

### 4.2 新增文件: `speech_act.py`

```python
class SpeechAct(Enum):
    MINIMAL_ACK = "minimal_ack"
    BRIEF_ANSWER = "brief_answer"
    ELABORATE_ANSWER = "elaborate_answer"
    SELF_DISCLOSE = "self_disclose"
    EMPATHIZE = "empathize"
    COMPLIMENT = "compliment"
    TEASE = "tease"
    EXTEND_TOPIC = "extend_topic"
    SHIFT_TOPIC = "shift_topic"
    CLOSE_TOPIC = "close_topic"
    QUESTION_BACK = "question_back"
    DISAGREE = "disagree"
    DEFLECT = "deflect"
    APOLOGIZE = "apologize"
    META_COMMENT = "meta_comment"
    SEEK_CLARIFICATION = "seek_clarification"

def select_speech_act(
    state: SessionState,
    user_msg: str,
    relationship_stage: str = "stranger"
) -> SpeechAct:
    """基于 Emotion → Mood → Trait → Relationship 的级联决策"""
    ...
```

### 4.3 新增文件: `language.py`

```python
@dataclass
class LanguageProfile:
    # 词汇层
    intensifier_rate: float = 0.3
    hedge_rate: float = 0.3
    positive_lexicon: float = 0.5
    negative_lexicon: float = 0.2
    emoji_rate: float = 0.2
    filler_rate: float = 0.15
    exclamation_rate: float = 0.3

    # 句法层
    avg_sentence_length: int = 25
    complexity: float = 0.5
    ellipsis_rate: float = 0.1
    question_rate: float = 0.2

    # 话语层
    response_length: str = "normal"
    politeness_strategy: str = "positive"
    turn_initiative: str = "neutral"
    self_disclosure_depth: float = 0.0
    humor_license: bool = False

def map_to_profile(
    speech_act: SpeechAct,
    emotion: Optional[Emotion],
    mood: Mood,
    trait: Trait,
    relationship_stage: str = "stranger"
) -> LanguageProfile:
    """将内部状态映射为量化的语言特征参数"""
    ...

def profile_to_prompt(profile: LanguageProfile) -> str:
    """将 LanguageProfile 编译为 30-40 token 的行为指令"""
    ...
```

### 4.4 Prompt 格式变化

v1:
```
[PERSONA_STATE]
当前情感基调:
  能量: ██████░░░░ 0.60 (中等能量...)
  ...
回复风格指引:
  - 回复简短...
[/PERSONA_STATE]
```
→ ~180 tokens

v2（Phase 3）:
```
[PERSONA]
语气: 温暖活泼，可以开玩笑
长度: 中等偏长，可以展开说
风格: 多用口语化表达，适当使用 emoji
克制: 不要过度亢奋，保持自然
[/PERSONA]
```
→ ~35 tokens，更接近行为指令，可解释性更好。

### 4.5 沉默权整合

v1 的 `silence.py` 移除，沉默逻辑分为两步：
1. **决策层**（Phase 1 `StateManager.silence_threshold`）：判断是否应沉默
2. **表达层**（Phase 3 `SpeechAct.MINIMAL_ACK`）：决定沉默时的具体话术

### 4.6 验收标准

- [ ] v1 五种表达方式被 Speech Act 完全替代，无功能回退
- [ ] Prompt 注入 ≤ 50 tokens（实测）
- [ ] 同一对话状态 + 不同 Speech Act → 明显不同的回复风格
- [ ] silence.py 安全移除

---

## 5. Phase 4 实施细节

### 5.1 目标

引入跨会话人际关系模型，让 AI 对新老用户表现不同。

### 5.2 新增文件: `relationship.py`

```python
@dataclass
class Relationship:
    user_id: str
    stage: str = "stranger"  # stranger | acquaintance | friend | close
    self_disclosure_depth: float = 0.0
    user_disclosure_depth: float = 0.0
    interaction_count: int = 0
    positive_exchanges: int = 0
    negative_exchanges: int = 0
    first_interaction: float = 0.0
    last_interaction: float = 0.0
    formality_match: float = 0.5

    @property
    def positive_ratio(self) -> float:
        total = self.positive_exchanges + self.negative_exchanges
        return self.positive_exchanges / total if total > 0 else 0.5

class RelationshipManager:
    def get(self, user_id: str) -> Relationship: ...
    def record_interaction(self, user_id: str, sentiment: float) -> None: ...
    def maybe_upgrade(self, user_id: str) -> None: ...
    def get_disclosure_limit(self, user_id: str) -> float: ...
    def reset(self, user_id: str) -> None: ...
```

### 5.3 数据存储

```json
{
  "sessions": { ... },
  "relationships": {
    "qq_official:webhook:123456789": {
      "stage": "friend",
      "self_disclosure_depth": 0.4,
      "user_disclosure_depth": 0.5,
      "interaction_count": 95,
      "positive_exchanges": 78,
      "negative_exchanges": 5,
      "first_interaction": 1718000000,
      "last_interaction": 1718700000,
      "formality_match": 0.6
    }
  }
}
```

### 5.4 hook 改动

```python
@filter.on_llm_request(priority=-100)
async def on_llm_request(self, event, req):
    sid = event.unified_msg_origin
    user_id = f"{event.get_platform_name()}:{event.get_sender_id()}"
    rel = self._rel_mgr.get(user_id)

    state = self._state_mgr.get_or_init(sid)
    self._state_mgr.drift_mood(sid)
    ...
    # 关系作为所有下游函数的参数
    sa = select_speech_act(state, user_msg, rel.stage)
    profile = map_to_profile(sa, state.emotion, state.mood, state.trait, rel.stage)
    block = profile_to_prompt(profile)
    ...
```

### 5.5 验收标准

- [ ] 与陌生人对话：回复偏礼貌、克制、不主动延伸
- [ ] 与朋友对话：回复更自然、允许玩笑、可自我表露
- [ ] 关系阶段按交互次数自动升级
- [ ] `interaction_count ≥ 20` 且正面比例 ≥ 0.7 → 升级到 acquaintance
- [ ] `/persona reset` 清除关系数据
- [ ] 每个 Phase 独立可用，Phase 4 完成后整体无回退

---

## 6. 测试策略

### 6.1 单元测试（每个 Phase 必须通过）

```python
# tests/test_state.py
def test_mood_ou_drift():
    """心境应该向基线回归"""
    mgr = StateManager(tmpdir)
    state = mgr.get_or_init("test")
    state.mood.valence = 0.1
    state.trait.mood_baseline["valence"] = 0.6
    # 模拟 24 小时漂移
    state.mood.updated_at = time.time() - 86400
    mgr.drift_mood("test")
    assert state.mood.valence > 0.3  # 应该向 0.6 回归

def test_emotion_decay():
    """情绪应该指数衰减"""
    ...

def test_appraisal_detection():
    """检测规则应该正确触发情绪"""
    ...

def test_emotion_pad_mapping():
    """每种离散情绪应映射到合理的 PAD 值"""
    ...
```

### 6.2 集成测试

```python
# tests/test_integration.py
def test_full_pipeline():
    """端到端：用户消息 → appraisal → emotion → speech_act → language → prompt"""
    mgr = StateManager(tmpdir)
    appraiser = AppraisalEngine()

    state = mgr.get_or_init("test")
    result = appraiser.evaluate("烦死了，又加班")
    assert result["emotion"] == EmotionLabel.ANGER

    mgr.trigger_emotion("test", result)
    assert state.emotion.primary == EmotionLabel.ANGER

    sa = select_speech_act(state, "烦死了，又加班")
    assert sa in [SpeechAct.BRIEF_ANSWER, SpeechAct.EMPATHIZE, SpeechAct.MINIMAL_ACK]

    profile = map_to_profile(sa, state.emotion, state.mood, state.trait)
    prompt = profile_to_prompt(profile)
    assert len(prompt) < 200  # token budget
    assert "愤怒" not in prompt  # 不能直接暴露内部状态给用户
```

### 6.3 手工测试清单

| 场景 | 操作 | 预期 |
|------|------|------|
| 新用户首聊 | 首次对话 | Trait 随机，Mood 中性，回复礼貌克制 |
| 连续多轮 | 聊 20 轮 | 耐心缓慢下降，回复逐渐变短 |
| 用户情绪感染 | 发"哈哈哈" | joy 触发，mood valence 小幅上升 |
| 情绪衰减 | 发"烦死了"后等 10 分钟 | anger 消失，mood 可能略微偏负 |
| 手动调 trait | `/persona trait extraversion 0.9` | 后续回复更活跃外向 |
| 话多模式 | `/persona talkative` | 修改 Trait + Mood，回复更长更主动 |
| 沉默触发 | 用户陈述完不提问 | 有一定概率发极简回应 |
| 老用户回归 | 隔天再聊 | 关系阶段保持，但 Mood 重新初始化 |

---

## 7. 迁移与兼容

### 7.1 states.json 版本管理

```json
{
  "_version": 2,
  "_migrated_from": 1,
  "sessions": { ... },
  "relationships": { ... }
}
```

### 7.2 向后兼容

- 所有 `/persona` 子命令保留，无破坏性移除（`/persona mode` 迁移到 Speech Act）
- `_conf_schema.json` 只增不减
- `metadata.yaml` 版本号更新为 `v2.0.0`

### 7.3 加载策略

- 开发阶段：在本地 workspace 用 Python 测试脚本验证
- 预发布：放入 AstrBot `plugins_disabled/`，手动 `mv` 到 `plugins/` 后重启测试
- 正式发布：直接放 `plugins/`，重启 AstrBot

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|:---:|------|
| Appraisal 规则覆盖率不足 | 中 | 大量消息不触发情绪 | 逐步扩充规则表；预备 fallback 情绪（由 mood 推导） |
| OU 参数调优困难 | 中 | 心境漂移太快/太慢 | 参数可配（`_conf_schema.json`），根据使用反馈迭代 |
| Speech Act 选择不自然 | 低 | 回复风格怪异 | Phase 3 先小范围测试，收集反馈后微调权重 |
| 关系模型隐私顾虑 | 低 | 用户不喜欢 AI 记住关系 | `/persona reset` 一键清除；关系数据不过度详细 |
| 复杂度失控 | 中 | 代码难维护 | 严格模块化，每个模块 ≤ 400 行；dataclass 不可变 |
| Phase 间耦合 | 低 | 后续 Phase 需要重构前面 | 每个 Phase 定义接口契约 (Python Protocol)，实现面向接口编程 |
