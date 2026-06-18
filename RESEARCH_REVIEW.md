# 情绪·逻辑·语言：相关研究综述

> 为 `astrbot_plugin_random_persona` 的下一步迭代提供理论支撑  
> 2026-06-18

---

## 目录

1. [情绪建模：从离散到连续](#1-情绪建模从离散到连续)
2. [人格与个体差异](#2-人格与个体差异)
3. [计算情绪模型：从符号到神经网络](#3-计算情绪模型从符号到神经网络)
4. [情绪动力学：时间尺度、调节与衰减](#4-情绪动力学时间尺度调节与衰减)
5. [语用学与会话理论](#5-语用学与会话理论)
6. [LLM + 情绪：当前技术路线](#6-llm--情绪当前技术路线)
7. [对当前插件的批判与改进路径](#7-对当前插件的批判与改进路径)

---

## 1. 情绪建模：从离散到连续

### 1.1 基本情绪理论 (Basic Emotion Theory)

**Ekman (1972, 1992)** 提出人类存在六种跨文化的 universal 基本情绪：**anger, disgust, fear, joy, sadness, surprise**。后扩展至 15+ 种。每种基本情绪对应：

- 独特的面部表情 (FACS 编码)
- 生理唤醒模式 (ANS 反应)
- 特定的行为倾向 (action tendency)

**对 AI 的意义**：基本情绪是离散的、可命名的、可被 prompt 指令直接调用的。例如"你现在感到悲伤"比"valence=0.2"更容易被 LLM 理解。

### 1.2 维度模型 (Dimensional Models)

**Russell's Circumplex Model (1980)**
- 两个正交维度：**valence (愉悦-不愉悦)** × **arousal (激活-未激活)**
- 所有情绪分布在环形空间中
- 例：兴奋 = 高愉悦 + 高激活；倦怠 = 低愉悦 + 低激活

**PAD 模型 (Mehrabian & Russell, 1974)**
- 三维：**Pleasure · Arousal · Dominance**
- Dominance（支配感）是关键补充：恐惧 = 不愉悦 + 高唤醒 + 低支配；愤怒 = 不愉悦 + 高唤醒 + 高支配
- 广泛用于 HCI、虚拟角色、产品设计

**当前插件的映射**：energy ≈ arousal, valence ≈ pleasure。**缺少 dominance**，无法区分恐惧 vs 愤怒、敬畏 vs 轻蔑。

### 1.3 认知评价理论 (Appraisal Theory)

**OCC 模型 (Ortony, Clore, Collins, 1988)** — 计算情感建模的基石理论。

核心思想：情绪不是直接由事件触发，而是由**认知评价 (appraisal)** 产生。评价维度包括：

| 评价维度 | 含义 | 例 |
|----------|------|-----|
| 目标相关性 (goal relevance) | 这件事与我的目标有关吗？ | 无关 → 无情绪 |
| 目标一致性 (goal congruence) | 促进还是阻碍我的目标？ | 促进 → joy, 阻碍 → distress |
| 预期性 (expectedness) | 符合预期吗？ | 意外 → surprise |
| 归因 (agency) | 谁造成的？ | 自己 → pride/shame, 他人 → anger/gratitude |
| 应对潜力 (coping potential) | 我能应对吗？ | 低 → fear/anxiety |
| 规范相容性 (norm compatibility) | 符合道德/社会规范吗？ | 违反 → anger/contempt |

**对 AI 的启示**：当前插件的状态漂移是**盲目的随机游走**，缺少 appraisal 驱动。真正的情绪应该有**因果事件**。

**Scherer's Component Process Model (2001, 2009)** — 更精细的序列评价模型：

```
新奇度检测 → 内在愉悦度 → 目标相关性 → 应对潜力 → 规范相容性
```

评价是**序列化、层级化**的，越高级的评价越依赖认知资源。

---

## 2. 人格与个体差异

### 2.1 Big Five / OCEAN (McCrae & Costa, 1987, 2003)

五个特质维度：

| 特质 | 高端 | 低端 |
|------|------|------|
| **Openness** 开放性 | 好奇、想象力、审美敏感 | 务实、惯例、偏好熟悉 |
| **Conscientiousness** 尽责性 | 自律、有条理、目标驱动 | 随意、灵活、可能散漫 |
| **Extraversion** 外向性 | 社交、活跃、寻求刺激 | 内向、独处、低调 |
| **Agreeableness** 宜人性 | 合作、同理心、信任 | 竞争、怀疑、直言 |
| **Neuroticism** 神经质 | 情绪不稳定、焦虑、易怒 | 情绪稳定、冷静、抗压 |

**对 AI 的意义**：
- Big Five 是**跨时间稳定**的基线，而情绪状态是**动态波动**的
- 人格 × 情绪交互：高神经质的人对负面事件反应更强（情绪反应性）、恢复更慢（情绪调节困难）
- 当前插件只有 openness 一个特质维度，**缺少神经质、宜人性、外向性、尽责性**

### 2.2 特质激活理论 (Trait Activation Theory, Tett & Guterman, 2000)

核心观点：**人格特质只在情境触发时才会表现出来**。

- 外向性在社交场合激活，独处时不表现
- 尽责性在任务压力下激活，放松时不表现
- 这意味着 AI 的"人格"应该在**不同对话场景下表现不同**

### 2.3 Eysenck's PEN 模型 (1967)

三维：**Psychoticism · Extraversion · Neuroticism**。
比 Big Five 更简约，但有生物学基础（皮层唤醒、边缘系统激活）。

---

## 3. 计算情绪模型：从符号到神经网络

### 3.1 ALMA (Gebhard, 2005)

**Layered Model of Affect** — 三层情感架构：

```
情绪 (Emotion) — 短时，由事件触发（秒-分钟级）
    ↕ 相互影响
心境 (Mood)    — 中时，缓慢漂移（小时-天级）
    ↕ 相互影响
人格 (Personality) — 长时稳定基线
```

关键机制：
- **情绪 → 心境**：多次同向情绪累积偏移心境（"今天遇到三件好事 → 心情好"）
- **心境 → 情绪**：心境影响事件的情绪反应强度（"心情好时更容易被逗笑"）
- **人格 → 情绪/心境**：Big Five 决定情绪反应的基线阈值

**这是当前插件最缺失的部分**。我们只有一个"状态"，没有情绪/心境/人格的分层。

### 3.2 EMA (Marsella & Gratch, 2009)

**Emotion and Adaptation model** — 基于 Smith & Lazarus (1990) 的评价理论。

核心循环：
```
事件 → 评价 → 情绪 → 应对策略 → 行为 → 新事件 → ...
```

EMA 的核心贡献是**应对策略 (coping strategies)** 建模：
- **问题聚焦应对**：改变情境（planning, seeking instrumental support）
- **情绪聚焦应对**：改变自己的感受（cognitive reappraisal, denial, venting）
- 策略选择受人格和当前情绪状态影响

**对 AI 的启示**：AI 应该有"应对策略"——不是被动地让情绪漂移，而是主动选择如何回应情境。

### 3.3 FAtiMA (Dias et al., 2014; Mascarenhas et al., 2022)

FAtiMA (FearNot! Affective Mind Architecture) — 面向交互叙事的情绪智能体架构。

特点：
- OCC 情绪评价
- **双向人际关系建模**（similarity, liking, dominance）
- 情绪影响行为选择
- 社会规范（cultural norms）约束行为

**人际关系维度**对对话 AI 特别有意义：
- AI 和用户之间的**关系阶段**（陌生人 → 熟人 → 朋友 → 亲密）
- 关系影响自我表露程度、玩笑容忍度、帮忙意愿等

### 3.4 WASABI (Becker-Asano, 2008)

**WASABI Affect Simulation for Agents with Believable Interactivity**

两层模型：
- **核心情感层**：PAD 三维度的连续动力学（微分方程驱动）
- **认知层**：基于 OCC 的离散情绪触发

强调**情绪的身体维度**（躯体标记），如呼吸、肌肉紧张。这在纯文本对话中难以直接体现，但可以通过**语言风格标记**间接模拟（如紧张 → 短句、重复用词、少用复杂结构）。

---

## 4. 情绪动力学：时间尺度、调节与衰减

### 4.1 心境 vs 情绪的区分

| 维度 | 情绪 (Emotion) | 心境 (Mood) |
|------|---------------|------------|
| 持续时间 | 秒-分钟 | 小时-天 |
| 强度 | 高 | 低-中 |
| 原因 | 有明确触发事件 | 无明确原因或累积原因 |
| 功能 | 中断当前活动、重新分配注意力 | 调整认知偏差、长期行为调节 |
| 神经基础 | 交感神经系统激活 | 神经递质（血清素、多巴胺） |

**当前插件的问题**：把情绪和心境混为一谈。"patience 持续下降"是心境变化，"用户说 哈哈哈 → valence +0.05"是情绪反应。应该分层。

### 4.2 情绪调节 (Emotion Regulation)

**Gross's Process Model (1998, 2015)** — 情绪调节的五个阶段：

```
情境选择 → 情境修正 → 注意部署 → 认知改变 → 反应调整
（最早）                                                （最晚）
```

关键调节策略：

| 策略 | 阶段 | 效果 | AI 中的模拟 |
|------|------|------|------------|
| **情境选择** | 最早 | 避开不愉快情境 | 话题切换 |
| **认知重评** (reappraisal) | 中期 | 改变对事件的解释 | "换个角度看…" |
| **表达抑制** (suppression) | 最晚 | 压抑外显情绪表达 | 回复变短、回避情绪词汇 |
| **分心** (distraction) | 早期 | 转移注意力 | 岔开话题 |
| **反刍** (rumination) | 晚期 | 反复思考负面事件 | 反复提及同一话题 |

**Gross 的关键发现**：
- 认知重评比表达抑制更健康（降低体验到的负面情绪 → 不会反弹）
- 表达抑制降低外在表达但**不降低**内在体验 → 还消耗认知资源
- **不同人格偏好不同策略**：高神经质偏 rumination，高开放性偏 reappraisal

**对插件的意义**：引入情绪调节策略维度，影响 AI 在被"惹到"或被负面情绪触发时的行为选择。

### 4.3 情绪衰减与习惯化

- **情绪的自然衰减**：未经维持的情绪在数分钟到数十分钟内自然消散
- **情感习惯化**：重复刺激导致的情绪反应递减（"第三十次听到同一个笑话就没感觉了"）
- **基线回归**：人在没有强烈刺激时，情绪向人格决定的基线回归（set-point theory, Headey & Wearing, 1989）

当前插件的 drift 只是加了随机噪声——应该建模为**向基线回归的均值回复过程** (Ornstein-Uhlenbeck process)：

```
dX = θ(μ - X)dt + σ dW
```
其中 μ 是人格决定的基线，θ 是回归强度，σ 是随机波动。

### 4.4 情绪感染 (Emotional Contagion)

**Hatfield, Cacioppo & Rapson (1993)**

- 人们无意识地模仿和同步他人的表情、声音、语言
- 导致情绪在对话者之间传播
- 机制：mimicry → feedback → contagion

**对 AI 的意义**：AI 应该对用户的情绪有**共鸣反应**——用户表达负面情绪时，AI 的 valence 应小幅下降（共情），但不能被完全吞噬（保持帮助性）。

---

## 5. 语用学与会话理论

### 5.1 Grice's Cooperative Principle (1975)

四项会话准则：

| 准则 | 含义 | 违反时的效果 |
|------|------|-------------|
| **量的准则** (Quantity) | 提供适量信息，不多不少 | 话太少→冷漠；话太多→啰嗦 |
| **质的准则** (Quality) | 说真话，不说无根据的话 | 违反→讽刺、夸张、修辞 |
| **关系准则** (Relation) | 说相关的话 | 岔开话题→回避或幽默 |
| **方式准则** (Manner) | 简洁有序，避免歧义 | 故意模糊→委婉、搪塞 |

**对 AI 的意义**：
- AI 的情绪状态应该影响对准则的遵守程度
- 低 patience → 违反量的准则（信息不足）→ 用户感知为"敷衍"
- 低 energy → 违反方式准则（表达散乱、用省略句）
- 高 openness → 遵守关系准则（主动关联话题）

### 5.2 Politeness Theory (Brown & Levinson, 1987)

核心概念：**面子 (face)**
- **积极面子**：希望被认可、被喜欢
- **消极面子**：希望不被干扰、行动自由

**面子威胁行为 (Face-Threatening Acts, FTAs)** 及其策略：

| 策略 | 直接程度 | 例子 |
|------|---------|------|
| Bald on-record | 最直接 | "帮我做这个" |
| Positive politeness | 温和 | "你超擅长这个的，能帮我看一下吗？" |
| Negative politeness | 间接 | "如果不麻烦的话，或许可以…" |
| Off-record | 暗示 | "这个好难啊…"（暗示需要帮助） |
| Don't do the FTA | 不说 | 沉默 |

**对 AI 的意义**：
- AI 的情绪状态应影响礼貌策略选择
- 低 valence + 低 energy → bald on-record（"嗯"、"好"）
- 高 valence + 高 energy → positive politeness（亲切热情）
- 高 openness → 更愿意使用 indirect strategies（委婉）

### 5.3 社会渗透理论 (Altman & Taylor, 1973)

**Social Penetration Theory** — 人际关系发展是**渐进、双向的自我表露过程**。

- **广度 (breadth)**：讨论的话题范围
- **深度 (depth)**：在每个话题上表露的亲密程度
- **洋葱模型**：表层（爱好、日常）→ 中层（目标、价值观）→ 核心层（恐惧、创伤）
- **互惠原则**：自我表露是双向的——"你告诉我你的，我告诉你我的"

**对 AI 的意义**：
- AI 和用户的**关系阶段**应该影响情感状态的基线
- 初次对话：高 patience、中 energy、中 valence → 礼貌但不过分热情
- 老用户：可以展示更多"个性和情绪波动"
- AI 的自我表露深度应与用户同步

### 5.4 话轮转换 (Turn-Taking)

**Sacks, Schegloff & Jefferson (1974)** — 对话中话轮转换的基本机制：

- **TRP (Transition Relevance Place)**：话轮可能交接的位置
- **话轮分配**：当前说话者选择下一位 vs 下一位自选
- **沉默的三种类型**：
  - **gap**：正常话轮转换间的小停顿（<1秒）
  - **lapse**：无人在 TRP 接话（>3秒，尴尬）
  - **pause**：同一说话者话轮内的停顿

**对 AI 的意义**：
- AI 的"沉默权"本质上是在 TRP 选择"不接话"
- 但人类不接话有不同的社交含义：思考中、不感兴趣、生气、不知道该说什么
- 需要**沉默类型标签**，而不仅仅是"沉默/不沉默"二分

### 5.5 语言风格匹配 (LSM) 与会话协调

**Communication Accommodation Theory (Giles, 1973, 2016)**
- **趋同 (convergence)**：调整自己的语言风格以更接近对方 → 寻求认同
- **趋异 (divergence)**：强调语言风格差异 → 区分身份
- **保持 (maintenance)**：不调整

**语言风格匹配 (Language Style Matching, Ireland et al., 2011)**
- 对话者在功能词使用上无意识同步（代词、介词、连词）
- LSM 程度预测关系质量和互动满意度
- 高 LSM → 更好的 rapport

**对 AI 的意义**：
- AI 应适度匹配用户的语域（口语化 ↔ 书面化、简洁 ↔ 详细）
- 趋同程度受 AI 情绪状态影响：高 valence → 更愿意趋同；低 valence → 更可能保持或趋异

---

## 6. LLM + 情绪：当前技术路线

### 6.1 Prompt-based 情绪注入（当前方案）

**方法**：在 system prompt 中注入情绪/人格指令。

**代表工作**：
- **EmotionPrompt (Li et al., 2023)**：在 prompt 中加入"这对我的职业发展非常重要"等情感刺激 → 提升 LLM 表现
- **Persona-based prompting**：定义角色人格 + 情绪状态 → LLM 角色扮演
- **Emotion-conditioned text generation (Goswamy et al., 2024)**：在 prompt 中指定目标情绪 → 生成带情绪色彩的文本
- **Sibyl (Wang et al., 2024)**：few-shot demonstration 选择影响 LLM 输出的人格和情绪风格

**优点**：实现简单、零训练成本、实时可调  
**缺点**：控制力弱（LLM 可能忽略指令）、稳定性差、缺乏因果模型

### 6.2 Instruction Tuning

在情绪标注数据上微调 LLM。

- **全量微调**：Emotional Qwen (Huang et al., 2024), PsyChat (Lai et al., 2024)
- **参数高效微调 (LoRA)**：Emotional LLaMA (Deng et al., 2024)

### 6.3 RLHF for Affect

通过人类偏好反馈或 AI 反馈优化情绪表达质量。

- RLHF 可以塑造 empathy, safety, emotional appropriateness
- **Constitutional AI 中的情绪规范**："不鼓励消极反刍"、"对脆弱用户保持支持但不过度共情"

### 6.4 多模态情绪

结合语音、面部表情、文本的情绪感知与生成：
- MER 2025 Challenge：多模态情绪识别 + 细粒度情绪 + 可解释性
- VoxCPM（我们在用的 TTS）：语音合成中引入情感控制

---

## 7. 对当前插件的批判与改进路径

### 7.1 当前设计的理论盲区

| 问题 | 现状 | 理论缺口 |
|------|------|----------|
| **维度不足** | 4维 (energy/valence/patience/openness) | 缺 dominance、缺离散情绪标签、缺 mood/emotion 分层 |
| **漂移无方向** | 纯随机游走 | 应建模为向基线回归的均值回复 + 事件驱动的跳跃 |
| **无情绪触发** | 只有机制性的 patience-0.02 | 缺少基于 appraisal 的情绪事件 |
| **无情绪调节** | 没有 reappraisal/suppression 等策略 | AI 应有应对策略选择 |
| **无关系模型** | 所有用户同等对待 | 应有关系阶段 + 自我表露深度追踪 |
| **无语言标记** | 笼统的"回复风格"描述 | 缺少具体的语言学特征映射 |
| **沉默无标签** | 沉默/不沉默二分 | 缺思考性沉默/情绪性沉默/社交性沉默的区分 |

### 7.2 推荐改进路线图

#### Phase 1：情绪分层（最小破坏性改动）

引入 ALMA 的三层结构：

```
[TRAITS]      基线人格 (跨会话稳定，可选随机初始化)
     ↓ 约束基线
[MOOD]        心境 (小时级，缓慢回归基线)
     ↓ 调节情绪反应强度
[EMOTION]     情绪 (由事件触发，分钟级衰减)
     ↓ 影响回复风格
[RESPONSE]    具体语言选择
```

**关键改动**：
- Traits → 当前 4 维 + Big Five 的部分维度（至少加 neuroticism）
- Mood → 对 emotion 的多次累积 + 向 trait 基线回归
- Emotion → 由用户输入中的 appraisal cues 触发

#### Phase 2：Appraisal 事件触发

不再盲目漂移。用户在 prompt 中的内容被解析为 appraisal 维度：

```
用户: "我又被老板批评了，烦死了"
  → novelty: medium
  → goal congruence: LOW (阻碍目标)
  → agency: OTHER (老板)
  → coping: LOW (表达了"烦死了")
  → 触发情绪: anger (agency=other) + sadness (coping=low) 
  → 情绪调节策略选择 (取决于 trait + mood)
```

#### Phase 3：语言特征映射

把抽象的情绪维度映射到可量化的语言学特征：

| 情绪状态 | 词汇层 | 句法层 | 话语层 |
|----------|--------|--------|--------|
| High energy | 强度副词(非常/超)、感叹号 | 短句、省略主语 | 主动提问、话题跳跃 |
| Low energy | 模糊限制语(可能/也许)、弱化 | 长停顿标记(…)、省略句 | 少主动发起 |
| High valence | 积极词汇、笑脸 emoji | 连贯叙述 | 自我表露、玩笑 |
| Low valence | 否定词、负面形容词 | 断裂句式 | 回避话题、最小化回应 |
| Low patience | 极简词汇 | 单句、片段 | 不展开、快速结束 |
| High openness | 丰富词汇、开放性问题 | 复杂句、嵌入从句 | 延伸话题、元评论 |

#### Phase 4：人际关系模型

```
Relationship State:
  stage: stranger | acquaintance | friend | close
  self_disclosure_depth: 0.0-1.0
  reciprocity_balance: -1.0 (user-led) to +1.0 (AI-led)
  interaction_count: int
  last_interaction: timestamp
```

关系阶段影响：
- 情绪表达的直接程度
- 沉默权的触发阈值
- 自我表露的深度
- 幽默和讽刺的使用许可

---

## 参考文献精选

1. **Ortony, A., Clore, G. L., & Collins, A.** (1988). *The Cognitive Structure of Emotions*. Cambridge University Press. — OCC 模型原典
2. **Scherer, K. R.** (2009). The dynamic architecture of emotion: Evidence for the component process model. *Cognition and Emotion*, 23(7), 1307-1351.
3. **Gebhard, P.** (2005). ALMA: A layered model of affect. *AAMAS 2005*.
4. **Marsella, S. C., & Gratch, J.** (2009). EMA: A process model of appraisal dynamics. *Cognitive Systems Research*, 10(1), 70-90.
5. **Gross, J. J.** (2015). Emotion regulation: Current status and future prospects. *Psychological Inquiry*, 26(1), 1-26.
6. **Brown, P., & Levinson, S. C.** (1987). *Politeness: Some Universals in Language Usage*. Cambridge University Press.
7. **Altman, I., & Taylor, D. A.** (1973). *Social Penetration: The Development of Interpersonal Relationships*. Holt, Rinehart & Winston.
8. **Grice, H. P.** (1975). Logic and conversation. In *Syntax and Semantics*, Vol. 3.
9. **Hatfield, E., Cacioppo, J. T., & Rapson, R. L.** (1993). Emotional contagion. *Current Directions in Psychological Science*, 2(3), 96-99.
10. **Ireland, M. E., et al.** (2011). Language style matching predicts relationship initiation and stability. *Psychological Science*, 22(1), 39-44.
11. **McCrae, R. R., & Costa, P. T.** (2003). *Personality in Adulthood: A Five-Factor Theory Perspective*. Guilford Press.
12. **Zhang, Y., et al.** (2024). Affective Computing in the Era of Large Language Models: A Survey. arXiv:2408.04638.
13. **Hegde, K., & Jayalath, H.** (2025). Emotions in the Loop: A Survey of Affective Computing for Emotional Support. arXiv:2505.01542.
14. **Smith, R., & Carette, J.** (2022). What Lies Beneath—A Survey of Affective Theory Use in Computational Models of Emotion. *IEEE TAC*.
