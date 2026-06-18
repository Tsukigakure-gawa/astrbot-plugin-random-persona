# 🎭 随机人格 v2 (Random Persona)

AstrBot 插件 — 六层认知-情感-社交架构，让 AI 回复呈现人类般的自然波动。

## 解决了什么问题

AI 聊天机器人在长期对话中容易表现三个"非人"特征：

1. **永远正面积极** — 不会低落、不会敷衍、不会不耐烦
2. **永远有回应** — 哪怕不需要回复也要硬找话题
3. **永远一种腔调** — 缺少人类自然切换表达方式的多样性

v2 通过**认知评价理论 + 社会渗透理论 + 语言特征量化映射**，让 AI 的回复风格随内部状态和对话关系动态变化。

---

## 架构

```
Relationship → Trait → Mood → Emotion → Speech Act → Language → Response
   关系阶段      人格基线    心境      事件触发      话语行为      语言参数
```

| 层 | 理论来源 | 时间尺度 | 说明 |
|----|---------|---------|------|
| **Relationship** | Social Penetration Theory (Altman & Taylor) | 百轮级 | 陌生人→熟人→朋友→亲密，自我表露互惠 |
| **Trait** | Big Five OCEAN (McCrae & Costa) | 跨会话稳定 | 开放性/尽责性/外向性/宜人性/神经质 |
| **Mood** | PAD 三维 (Mehrabian & Russell) | 小时级 | 愉悦度/唤醒度/支配感，OU 均值回复过程 |
| **Emotion** | OCC 评价理论 (Ortony, Clore, Collins) | 分钟级 | 11 种离散情绪，Appraisal 事件→指数衰减 |
| **Speech Act** | 话语行为理论 (Searle / Austin) | 每轮 | 17 种话语行为：最小回应/展开/延伸/回避/反问… |
| **Language** | 社会语言学 + 语用学 | 每轮 | 词汇层/句法层/话语层 16 参数量化映射 |

---

## 核心机制

### Appraisal 情绪触发（零 LLM 成本）

用户消息 → 规则引擎检测认知评价线索 → 触发离散情绪：

| 用户说 | 检测线索 | 触发情绪 | 半衰期 |
|--------|---------|---------|--------|
| "烦死了/气死我了" | 目标阻碍 + 他人归因 | anger | 10 min |
| "哈哈哈/太棒了" | 目标促进 + 意外 | joy | 3 min |
| "难过/想哭/好累" | 目标阻碍 + 应对力低 | sadness | 15 min |
| "怎么办/好怕" | 应对力低 + 高相关 | fear | 5 min |
| "卧槽/不是吧" | 意外 | surprise | 1 min |
| "恶心/下头" | 规范违反 | disgust | 7.5 min |
| "谢谢/多亏你" | 他人归因 + 目标促进 | gratitude | 5 min |

### 情绪调节策略

触发情绪后，根据人格选择应对策略，影响后续话语行为：

| 策略 | 效果 | 高概率人格 |
|------|------|-----------|
| suppression | 抑制外显，回复简短回避 | 高神经质 |
| reappraisal | 换角度理解，恢复理性 | 高开放性+高宜人性 |
| rumination | 反复表达，回复拉长 | 高神经质 |
| controlled_expression | 克制但有态度 | 高尽责性 |
| amplify | 放大情绪表达 | 高外向性 |

### 沉默权（非机械式）

三条件判断 + 关系调节：

- **用户没提问** + AI 上轮已充分回复（>150字）
- **耐心极低**（<0.2）
- 低能+偏冷 → 30% 概率

关系调节：陌生人少沉默（礼貌），熟人可自然沉默。

### 语言特征量化映射

抽象情绪 → 16 个可验证参数 → ~35 token 行为指令：

```
[PERSONA]
语气: 克制、有力
长度: 简洁
风格: 短句为主
克制: 少用感叹号、不用夸张修辞
[/PERSONA]
```

vs v1 的 ~180 token 状态表格，节省 80%。

---

## 安装

将插件文件夹放入 AstrBot `data/plugins/` 目录，重启容器。

```bash
cp -r astrbot_plugin_random_persona /home/admin/astrbot/data/plugins/
docker restart astrbot
```

## 命令

| 命令 | 功能 |
|------|------|
| `/persona` | 查看完整状态（Trait / Mood / Emotion / 关系） |
| `/persona random` | 随机重置人格 + 心境 |
| `/persona chill` | 低外向 + 低唤醒 + 偏冷 |
| `/persona warm` | 高外向 + 高宜人 + 偏暖 |
| `/persona talkative` | 高外向 + 高开放 + 话多 |
| `/persona quiet` | 低外向 + 低唤醒 + 少说 |
| `/persona trait <维度> <0-1>` | 调节 OCEAN 特质 |
| `/persona emotion <标签>` | 手动触发情绪 |
| `/persona reset` | 重置人格 + 清除关系数据 |
| `/persona off` / `on` | 开关 |

**情绪标签：** `joy` `anger` `sadness` `fear` `surprise` `disgust` `trust` `anticipation` `guilt` `gratitude` `hurt`

**特质维度：** `openness` `conscientiousness` `extraversion` `agreeableness` `neuroticism`

## 配置

在 AstrBot WebUI → 插件管理 → 随机人格 → 配置：

| 配置项 | 选项 | 默认值 | 说明 |
|--------|------|--------|------|
| 沉默模式 | 短回应 / 完全不回 | 短回应 | 触发沉默权时：发极简回复或不发 |

## 对比 v1

| 维度 | v1.0 | v2.0 |
|------|------|------|
| 状态层数 | 1 (扁平 4 维) | 3 (Trait/Mood/Emotion) + Relationship |
| 漂移方式 | 纯随机游走 | OU 均值回复 |
| 情绪触发 | 无 | 11 种离散情绪，规则引擎 |
| 情绪调节 | 无 | 8 种策略，受人格影响 |
| 话语行为 | 5 种表达方式（粗略） | 17 种 Speech Act |
| 语言映射 | 模糊风格描述 | 16 参数量化（词法/句法/话语） |
| 关系模型 | 无 | 4 阶段社会渗透 + 自我表露洋葱 |
| Prompt 注入 | ~180 tokens | ~35 tokens |
| Token 成本 | ~180/轮 | ~35/轮 (↓80%) |
| 代码规模 | 777 行 (6 文件) | 2,019 行 (7 文件) |

## 设计原则

1. **不假装有意识** — 情绪是风格模拟，不声称主观体验
2. **核心能力不降级** — 实质问题时准确性优先于风格
3. **适度不可预测** — 模拟自然的不一致性，不牺牲可用性
4. **沉默是表态，不是故障** — 用极简回应清晰表达"已读但不必多说"
5. **零额外 LLM 成本** — 所有检测/映射均为规则引擎

## 参考资料

- OCC 评价理论 (Ortony, Clore, Collins, 1988)
- PAD 情绪维度模型 (Mehrabian & Russell, 1974)
- ALMA 三层情感架构 (Gebhard, 2005)
- EMA 情绪与适应模型 (Marsella & Gratch, 2009)
- Gross 情绪调节过程模型 (1998, 2015)
- 社会渗透理论 (Altman & Taylor, 1973)
- 面子协商理论 (Brown & Levinson, 1987)
- Big Five 人格理论 (McCrae & Costa, 2003)
- LLM 时代的情感计算综述 (Zhang et al., 2024, arXiv:2408.04638)

完整综述见仓库内 `RESEARCH_REVIEW.md`。

## License

MIT
