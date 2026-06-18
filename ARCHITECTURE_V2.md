# 🎭 random-persona v2 — 架构设计文档

> 从扁平 4 维随机游走 → 四层认知-情感-社交架构  
> 2026-06-18

---

## 目录

1. [总体架构](#1-总体架构)
2. [Phase 1: 情绪分层 (ALMA三层)](#2-phase-1-情绪分层)
3. [Phase 2: Appraisal 事件触发](#3-phase-2-appraisal-事件触发)
4. [Phase 3: 话语行为 + 语言特征映射](#4-phase-3-话语行为--语言特征映射)
5. [Phase 4: 人际关系模型](#5-phase-4-人际关系模型)
6. [实施顺序与依赖](#6-实施顺序与依赖)

---

## 1. 总体架构

```
┌─────────────────────────────────────────────────────┐
│                    RELATIONSHIP                      │
│  关系阶段 · 自我表露深度 · 互惠平衡                  │
│  (跨会话持久化，缓慢演进)                            │
└──────────────────┬──────────────────────────────────┘
                   │ 调节情绪表达阈值 & 语言风格
                   ▼
┌─────────────────────────────────────────────────────┐
│                     TRAITS                           │
│  OCEAN 基线人格 (跨会话持久化，极缓变)               │
│  决定情绪基线、调节偏好、表达风格                     │
└──────────────────┬──────────────────────────────────┘
                   │ 设定基线，约束回归目标
                   ▼
┌─────────────────────────────────────────────────────┐
│                      MOOD                            │
│  心境 (会话级，小时级衰减，向 Trait 基线回归)        │
│  调节情绪反应强度，影响认知偏差                       │
└──────────────────┬──────────────────────────────────┘
                   │ 调节情绪触发阈值 & 反应强度
                   ▼
┌─────────────────────────────────────────────────────┐
│                    EMOTION                           │
│  情绪 (由 Appraisal 事件触发，分钟级衰减)            │
│  离散情绪标签 + 连续维度                              │
└──────────────────┬──────────────────────────────────┘
                   │ 选择 Speech Act
                   ▼
┌─────────────────────────────────────────────────────┐
│                   SPEECH ACT                         │
│  话语行为选择 (表达·询问·回避·延伸·最小化…)          │
│  受 Emotion + Mood + Trait + Relationship 共同影响   │
└──────────────────┬──────────────────────────────────┘
                   │ 参数化语言生成
                   ▼
┌─────────────────────────────────────────────────────┐
│                LANGUAGE MAPPING                      │
│  词汇层 (强度副词/模糊限制语/情感词…)               │
│  句法层 (句长/复杂度/省略…)                          │
│  话语层 (话题管理/话轮行为/礼貌策略…)               │
└──────────────────┬──────────────────────────────────┘
                   │ 注入 system prompt
                   ▼
┌─────────────────────────────────────────────────────┐
│                   LLM RESPONSE                       │
└─────────────────────────────────────────────────────┘
```

**关键改进 vs v1.0：**
- 情绪不再是无方向的随机游走，而是**事件触发 → 衰减 → 基线回归**
- 引入**离散情绪标签**（愤怒、喜悦、悲伤等），不只是连续维度
- **Speech Act** 填补了情绪 → 语言之间的逻辑空白
- **关系模型**让同样的情绪对熟人和陌生人表现不同
- **语言映射**让"语气温暖"有可验证的语言学参数

---

## 2. Phase 1: 情绪分层

### 2.1 三层数据模型

#### Layer 1: TRAITS（基线人格）

```python
Traits = {
    # OCEAN 五因素 (0.0-1.0)
    "openness":          0.55,   # 开放性：好奇/审美 → 务实/常规
    "conscientiousness": 0.60,   # 尽责性：自律/条理 → 随意/灵活
    "extraversion":      0.45,   # 外向性：社交/活跃 → 内向/独处
    "agreeableness":     0.65,   # 宜人性：合作/共情 → 竞争/怀疑
    "neuroticism":       0.40,   # 神经质：焦虑/易怒 → 稳定/冷静

    # 会话初始化参数
    "mood_baseline": {            # 心境回归目标 (由 OCEAN 推导)
        "valence":  0.55,
        "arousal":  0.50,
        "dominance": 0.55,
    },
    "regulation_style": {         # 情绪调节偏好
        "reappraisal":   0.55,    # 认知重评倾向
        "suppression":   0.30,    # 表达抑制倾向
        "rumination":    0.25,    # 反刍倾向
    },
}
```

**特性：**
- 跨会话持久化，极缓慢演变（数十次对话后才可能微调）
- 对话开始时随机初始化（保留 v1 的随机性），之后稳定
- OCEAN 推导出 mood_baseline 和 regulation_style
- 用户可通过 `/persona trait <维度> <值>` 手动调

#### Layer 2: MOOD（心境）

```python
Mood = {
    # PAD 三维 (0.0-1.0)
    "valence":   0.55,    # 愉悦度
    "arousal":   0.38,    # 唤醒度
    "dominance": 0.50,    # 支配感

    "updated_at": 1718700000,
}
```

**核心动力学：Ornstein-Uhlenbeck 均值回复过程**

```
dValence  = θ_v × (baseline_v - valence)  × dt  +  σ_v × dW_v
dArousal  = θ_a × (baseline_a - arousal)  × dt  +  σ_a × dW_a
dDominance = θ_d × (baseline_d - dominance) × dt + σ_d × dW_d
```

其中：
- `baseline_*` 来自 Traits.mood_baseline
- `θ`（回归强度）≈ 0.1-0.3，对应几小时到半天回归基线
- `σ`（随机波动）≈ 0.02-0.05
- `dt` = 距上次更新的时间间隔

**交互影响（情绪累积 → 心境偏移）：**
- 连续多次同向情绪 → 心境向该方向偏移（如三次愤怒 → valence -0.08）
- 情绪强度越高，偏移越大
- 偏移后心境仍向 Trait 基线回归，但回归速率减慢

#### Layer 3: EMOTION（情绪）

```python
Emotion = {
    # 离散标签 (可多个，取主导)
    "primary":    None | "joy" | "sadness" | "anger" | "fear" |
                  "surprise" | "disgust" | "trust" | "anticipation",
    "secondary":  None | str,

    # PAD 三维 (事件触发的瞬时值)
    "valence":    0.0,
    "arousal":    0.0,
    "dominance":  0.0,

    # 元信息
    "trigger":    "user_insult",     # 触发原因标签
    "intensity":  0.0,               # 强度 (0.0-1.0)
    "started_at": 1718700000,
    "half_life":  300,               # 半衰期 (秒)，默认 5 分钟
}
```

**情绪生命周期：**
```
Event → Appraisal → Emotion 触发 (离散标签 + PAD + intensity)
  → 自然衰减 (exponential decay, half-life 决定)
  → 低于阈值后重置为 None
```

**情绪 → 心境反馈：**
- 情绪衰减到 0 时，将其 PAD 值按衰减积分的一部分加到 Mood
- 多次同向情绪累积改变心境

### 2.2 v1 维度到 v2 的映射

| v1 维度 | v2 位置 | 说明 |
|---------|---------|------|
| energy | Trait.extraversion + Mood.arousal | 拆分为特质基线 + 当前唤醒度 |
| valence | Mood.valence + Emotion.valence | 心境愉悦度 + 瞬时情绪效价 |
| patience | 计算属性：由 Trait.neuroticism + Mood + 会话长度推导 | 不是基础维度 |
| openness | Trait.openness | 升为稳定特质 |
| mode (五种表达) | Speech Act 层替代 | 不再是独立维度 |

### 2.3 计算属性（由基础维度推导）

```python
def compute_patience(trait, mood, session_msg_count):
    """耐心 = 特质基线 - 心境负面偏移 - 会话疲劳"""
    base = 1.0 - trait["neuroticism"] * 0.5     # 高神经质 → 低基线耐心
    mood_penalty = max(0, 0.5 - mood["valence"]) * 0.4
    fatigue = min(0.4, session_msg_count * 0.02)
    return max(0.0, min(1.0, base - mood_penalty - fatigue))

def compute_silence_threshold(trait, mood, relationship):
    """沉默权触发阈值（越高越容易沉默）"""
    ...
```

---

## 3. Phase 2: Appraisal 事件触发

### 3.1 Appraisal 引擎

从用户消息中检测评价线索，映射到离散情绪。**不用 LLM 做**（零额外 token 成本），用规则引擎 + 关键词/模式匹配。

```python
AppraisalResult = {
    "goal_relevance":    0.0-1.0,   # 与目标相关吗？
    "goal_congruence":   -1.0~1.0,  # 促进 (+) 还是阻碍 (-) 目标？
    "expectedness":      0.0-1.0,   # 意外程度
    "agency":            "self" | "other" | "circumstance" | "none",
    "coping_potential":  0.0-1.0,   # 我能应对吗？
    "norm_compatibility": -1.0~1.0, # 符合 (+) 还是违反 (-) 规范？
}
```

### 3.2 检测规则示例

| 用户消息模式 | Appraisal 推断 | 触发情绪 |
|-------------|---------------|---------|
| "烦死了/气死我了/无语" | goal_congruence=-0.7, agency=other | anger |
| "好开心/太棒了/哈哈哈" | goal_congruence=+0.7, agency=circumstance | joy |
| "怎么办/我好怕/不敢" | coping=-0.5, goal_relevance=0.8 | fear/anxiety |
| "对不起/我的错/怪我" | agency=self, goal_congruence=-0.6 | sadness/guilt |
| "卧槽/真的假的/不是吧" | expectedness=-0.6 | surprise |
| "你真好/谢谢你/多亏你" | agency=other, goal_congruence=+0.8 | gratitude |
| 批评/攻击 AI 本身 | agency=other, goal_congruence=-0.5, target=AI | hurt/frustration |
| 用户自我贬低 | agency=self, coping=-0.4 | concern+compassion |

### 3.3 情绪衰减

```
intensity(t) = intensity_0 × 2^(-t / half_life)
```

不同情绪有不同的半衰期：
- anger: 10 分钟（来得快去得慢）
- joy: 3 分钟
- surprise: 1 分钟
- sadness: 15 分钟
- fear: 5 分钟

情绪低于 `intensity < 0.05` 时重置为 None。

### 3.4 调节策略选择

**同时受 Trait 和当前状态影响：**

| 被触发情绪 | 可选策略 | 选择权重影响因素 |
|-----------|---------|----------------|
| anger | suppression / reappraisal / controlled_expression | 高 agreeableness → reappraisal；高 neuroticism → suppression |
| sadness | acceptance / rumination / distraction | 高 neuroticism → rumination；高 openness → acceptance |
| joy | amplification / sharing / moderation | 高 extraversion → sharing；陌生人 → moderation |
| fear | seeking_support / avoidance / reappraisal | 高 dominance → reappraisal；低 dominance → seeking_support |

**策略会反馈到 speech act 选择。** 例如 reappraisal → "换个角度想…"

---

## 4. Phase 3: 话语行为 + 语言特征映射

### 4.1 Speech Act 层

**为什么需要它：** 情绪不能直接映射到语言。同样的"愤怒"，可能选择讽刺（议论+隐晦）、直接指责（说明+攻击性）、闭口不谈（最小化+回避）。Speech Act 是情绪和语言之间的决策层。

```python
SpeechAct = enum {
    # 回应类
    MINIMAL_ACK,         # 最小确认 ("嗯", "好")
    ELABORATE_ANSWER,    # 展开回答
    BRIEF_ANSWER,        # 简洁回答

    # 社会关系类
    SELF_DISCLOSE,       # 自我表露
    EMPATHIZE,           # 共情
    COMPLIMENT,          # 称赞
    TEASE,               # 调侃

    # 话题管理类
    EXTEND_TOPIC,        # 延伸话题
    SHIFT_TOPIC,         # 转移话题
    RETURN_TOPIC,        # 回到原话题
    CLOSE_TOPIC,         # 结束话题

    # 对抗/防御类
    QUESTION_BACK,       # 反问
    DISAGREE,            # 表达异议
    DEFLECT,             # 回避/搪塞
    APOLOGIZE,           # 道歉

    # 元对话类
    META_COMMENT,        # 元评论 ("这个问题挺有意思的")
    SEEK_CLARIFICATION,  # 请求澄清
}
```

**Speech Act 选择函数：**

```python
def select_speech_act(emotion, mood, trait, relationship) -> SpeechAct:
    # 沉默权检查 → MINIMAL_ACK
    if should_silence(...):
        return MINIMAL_ACK

    # 情绪驱动的基线
    if emotion and emotion.primary == "anger":
        candidates = [BRIEF_ANSWER, QUESTION_BACK, DEFLECT, DISAGREE]
    elif emotion and emotion.primary == "joy":
        candidates = [ELABORATE_ANSWER, EXTEND_TOPIC, SELF_DISCLOSE, TEASE]
    ...

    # Trait 过滤
    if trait["agreeableness"] < 0.3:
        candidates.discard(COMPLIMENT)
        candidates.discard(EMPATHIZE)

    # Relationship 调节
    if relationship.stage < "friend":
        candidates.discard(TEASE)
        candidates.discard(SELF_DISCLOSE)

    # 加权随机选择
    return weighted_choice(candidates, weights)
```

### 4.2 语言特征参数

Speech Act 选择后，将内部状态参数化为可注入 prompt 的语言特征：

```python
LanguageProfile = {
    # ── 词汇层 ──
    "intensifier_rate":    0.0-1.0,  # 强度副词频率 ("非常"/"超级")
    "hedge_rate":          0.0-1.0,  # 模糊限制语频率 ("可能"/"也许")
    "positive_lexicon":    0.0-1.0,  # 积极词汇偏向
    "negative_lexicon":    0.0-1.0,  # 消极词汇偏向
    "emoji_rate":          0.0-1.0,  # Emoji 使用频率
    "filler_rate":         0.0-1.0,  # 语气词频率 ("嗯"/"啊"/"嘛")
    "exclamation_rate":    0.0-1.0,  # 感叹号频率

    # ── 句法层 ──
    "avg_sentence_length": 5-50,     # 平均句子长度 (字)
    "complexity":          0.0-1.0,  # 句法复杂度 (嵌套/从句)
    "ellipsis_rate":       0.0-1.0,  # 省略句比例
    "question_rate":       0.0-1.0,  # 问句比例

    # ── 话语层 ──
    "response_length":     "minimal" | "brief" | "normal" | "elaborate",
    "politeness_strategy": "bald" | "positive" | "negative" | "off_record",
    "turn_initiative":     "passive" | "neutral" | "active",
    "self_disclosure_depth": 0.0-1.0,
    "humor_license":       False,    # 是否允许开玩笑
}
```

### 4.3 情绪 → LanguageProfile 映射规则（示例）

| 状态 | intensifier | hedge | sentence_len | politeness |
|------|:----------:|:-----:|:----------:|------------|
| anger + controlled_expression | 0.6 | 0.1 | brief | bald |
| joy + amplify | 0.8 | 0.1 | normal | positive |
| sadness + acceptance | 0.3 | 0.5 | brief | negative |
| fear + seeking_support | 0.4 | 0.6 | normal | negative |
| 低 energy + 低 valence | 0.1 | 0.3 | minimal | bald |
| 高 energy + 高 valence + 朋友 | 0.7 | 0.1 | elaborate | positive |

**关键点：Relationship 作为调节参数。** 同一情绪 + Speech Act，对陌生人礼貌策略升一档，对熟人 humor_license 可以开。

### 4.4 生成最终的 System Prompt 注入

不再注入复杂的 `[PERSONA_STATE]` 表格，改为**行为指令**：

```
[PERSONA]
语气: 偏冷静，带克制的不满
表达: 简洁反问，点到即止
长度: 一两句话
禁止: 感叹号、过度情绪化词汇、主动延伸话题
[/PERSONA]
```

≈ 30-40 tokens，比 v1 的完整状态块更简洁，但参数化程度更高。

---

## 5. Phase 4: 人际关系模型

### 5.1 关系状态

```python
Relationship = {
    "stage":  "stranger" | "acquaintance" | "friend" | "close",

    # 社会渗透维度
    "self_disclosure_depth":  0.0-1.0,   # AI 已表露的深度
    "user_disclosure_depth":  0.0-1.0,   # 用户已表露的深度 (观测值)

    # 社交指标
    "interaction_count":      0,          # 总交互轮数
    "first_interaction":      1718700000,
    "last_interaction":       1718700000,
    "positive_exchanges":     0,          # 正向交互计数
    "negative_exchanges":     0,          # 负向交互计数

    # 风格同步
    "formality_match":        0.5,        # 语域匹配程度
}
```

### 5.2 阶段升级条件

| 阶段 | 条件 |
|------|------|
| stranger | 默认 |
| → acquaintance | interaction ≥ 20 且 positive_ratio ≥ 0.7 |
| → friend | interaction ≥ 80 且 self_disclosure 双向发生 |
| → close | 用户主动深层次表露 + AI 回应共情 + 长期互动 |

**不会自动降级。** 除非用户明确重置。

### 5.3 关系影响矩阵

| 维度 | stranger | acquaintance | friend | close |
|------|----------|-------------|--------|-------|
| 沉默权阈值 | 低（多回） | 中 | 中高 | 高（可以自然沉默） |
| humor_license | ❌ | ❌ | ✅ | ✅ |
| tease_allowed | ❌ | ❌ | 谨慎 | ✅ |
| self_disclosure_max | 0.1 | 0.3 | 0.6 | 0.9 |
| 情绪表达直接度 | 克制 | 中性 | 适度 | 自然 |
| 礼貌策略 | negative | positive | positive | bald 可接受 |
| 追问/延伸频率 | 低 | 中 | 高 | 自然 |

### 5.4 自我表露的洋葱模型

```
表层 (0.0-0.2): 爱好、日常、天气
浅层 (0.2-0.4): 观点、偏好、小故事
中层 (0.4-0.6): 价值观、目标、困惑
深层 (0.6-0.8): 弱点、失败经历、不安
核心 (0.8-1.0): 创伤、恐惧、核心信念
```

AI 的表露深度不应超过用户已表露的深度 + 0.2（互惠但不逾越）。

---

## 6. 实施顺序与依赖

```
Phase 1 ──────► Phase 2 ──────► Phase 3 ──────► Phase 4
(分层模型)      (appraisal)     (speech act     (关系模型)
                 依赖: Phase1     +语言映射)      依赖: Phase3
                                 依赖: Phase1+2
```

### Phase 1 实施要点

**改动文件：** `state.py`（重写） + `main.py`（适配） + `prompt.py`（适配）+ `silence.py`（适配）

**破坏性改动：**
- `state.py` 现有 170 行 → 预计 ~350 行（三层结构 + OU 过程 + 计算属性）
- 存储格式 `states.json` 需要 migration
- `main.py` 的 hook 逻辑从 "get state → drift → inject" 变为 "get state → mood_drift → appraisal → emotion → inject"
- 命令接口新增 `/persona trait`，`/persona` 输出格式大变

**不破坏的：**
- `_conf_schema.json` 基本不变
- `metadata.yaml` 改版本号
- `/persona` 命令族保留，增强

### Phase 2 实施要点

**新增文件：** `appraisal.py`（评价引擎 + 检测规则 + 情绪映射）

**改动文件：** `main.py`（hook 中插入 appraisal 步骤）+ `state.py`（新增 Emotion 的衰减更新）

**核心挑战：** 规则引擎的覆盖率。中文情绪检测模式需要较丰富的词典。初期可以覆盖 60-70% 常见表达，后续迭代补充。

**不引入 LLM 做 appraisal。** 零 token 成本是硬约束。

### Phase 3 实施要点

**新增文件：** `speech_act.py`（Speech Act 选择器）+ `language_profile.py`（语言特征参数映射）

**改动文件：** `prompt.py`（改为生成行为指令而非状态表格）+ `silence.py`（整合到 Speech Act 层）

### Phase 4 实施要点

**新增文件：** `relationship.py`（关系状态管理 + 表露互惠逻辑）

**改动文件：** `speech_act.py`（关系参数输入）+ `language_profile.py`（关系调节）+ `main.py`（关系状态更新）

---

## 附录 A: 与 v1.0 的对比

| 维度 | v1.0 | v2 目标 |
|------|------|---------|
| 状态层数 | 1 (扁平4维) | 3 (Trait/Mood/Emotion) |
| 漂移方式 | 纯随机游走 | OU 均值回复 + 事件驱动 |
| 情绪触发 | 无 | Appraisal 引擎 |
| 情绪类型 | 无离散标签 | 9 种基本情绪 |
| 话语层 | 无 | Speech Act + LanguageProfile |
| 关系模型 | 无 | 4 阶段社会渗透 |
| Prompt 注入 | ~180 token 状态表 | ~35 token 行为指令 |
| token 开销 | ~180/轮 | ~35/轮 (↓80%) |
