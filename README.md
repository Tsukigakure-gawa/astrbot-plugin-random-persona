# 🎭 随机人格 v2 (Random Persona)

AstrBot 插件 — 六层认知-情感-社交架构，让 AI 回复不再千篇一律。

## 架构

```
Relationship → Trait → Mood → Emotion → Speech Act → Language → Response
   (关系阶段)   (人格基线)  (心境)    (事件触发)   (话语行为)    (语言参数)
```

### 六层详解

| 层 | 解释 | 时间尺度 |
|----|------|---------|
| **Relationship** | 社会渗透：陌生人→熟人→朋友→亲密 | 跨会话，数百轮 |
| **Trait** | OCEAN 五因素基线人格 | 跨会话稳定 |
| **Mood** | PAD 三维心境，OU 均值回复 | 小时级 |
| **Emotion** | Appraisal 事件触发，指数衰减 | 分钟级 |
| **Speech Act** | 17 种话语行为选择 | 每轮 |
| **Language** | 16 参数语言特征 → ~35t prompt 注入 | 每轮 |

### 关键机制

- **Appraisal 引擎** — 从用户消息检测认知评价线索，触发 11 种离散情绪（规则引擎，零 LLM 成本）
- **情绪调节** — 8 种调节策略（suppression/reappraisal/rumination 等），受 Trait + Mood 影响
- **语言特征映射** — 情绪/人格 → 词汇层(7参数) / 句法层(4参数) / 话语层(5参数)
- **关系模型** — 社会渗透 + 自我表露洋葱 + 互惠原则
- **沉默权** — 关系调节：陌生人少沉默，熟人可自然沉默

## 安装

将插件文件夹放入 AstrBot 的 `data/plugins/` 目录。

## 命令

| 命令 | 功能 |
|------|------|
| `/persona` | 查看三层状态 (Trait / Mood / Emotion) |
| `/persona random` | 随机重置 |
| `/persona chill` / `warm` / `talkative` / `quiet` | 快速模式 |
| `/persona trait <维度> <值>` | 手动调 OCEAN 特质 |
| `/persona emotion <标签>` | 手动触发情绪 |
| `/persona reset` | 重置人格 + 关系 |
| `/persona off` / `on` | 开关 |

## 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 沉默模式 | 触发沉默权时：短回应 / 完全不回 | 短回应 |

## 对比 v1

| 维度 | v1.0 | v2.0 |
|------|------|------|
| 状态层数 | 1 (扁平4维) | 3 (Trait/Mood/Emotion) |
| 漂移方式 | 纯随机游走 | OU 均值回复 |
| 情绪触发 | 无 | 11 种离散情绪，规则引擎 |
| 话语层 | 无 | 17 种 Speech Act |
| 语言映射 | 模糊风格描述 | 16 参数量化 |
| 关系模型 | 无 | 4 阶段社会渗透 |
| Prompt 注入 | ~180 tokens | ~35 tokens |

## License

MIT
