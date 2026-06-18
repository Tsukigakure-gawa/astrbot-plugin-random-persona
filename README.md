# 🎭 随机人格 v2 (Random Persona)

AI 回复风格动态引擎 — 六层认知-情感-社交架构，让 LLM 回复呈现人类般的自然波动。

## 解决了什么

AI 聊天机器人在长期对话中容易表现三个"非人"特征：

1. **永远正面积极** — 不会低落、不会敷衍、不会不耐烦
2. **永远有回应** — 哪怕不需要回复也要硬找话题
3. **永远一种腔调** — 缺少人类自然切换表达方式的多样性

## 架构

```
Relationship → Trait → Mood → Emotion → Speech Act → Language → Response
   关系阶段      人格基线    心境      事件触发      话语行为      语言参数
```

六层全部基于规则引擎 + PAD 情感词库驱动，零额外 LLM token 成本。

| 层 | 理论来源 | 说明 |
|----|---------|------|
| Relationship | Social Penetration Theory | 4 阶段：陌生人→熟人→朋友→亲密 |
| Trait | Big Five OCEAN | 开放性/尽责性/外向性/宜人性/神经质 |
| Mood | PAD 三维 + OU 过程 | 愉悦度/唤醒度/支配感，向基线回归 |
| Emotion | OCC 评价理论 | 11 种离散情绪，Appraisal 事件触发→指数衰减 |
| Speech Act | 话语行为理论 | 17 种话语行为：展开/延伸/回避/反问/共情… |
| Language | 1600 词 PAD 词库 | 词法/句法/话语 16 参数量化→~50t prompt 注入 |

## 项目结构

```
random-persona-mcp/         ← MCP 服务端（独立进程，提供工具）
astrbot_plugin_random_persona/  ← AstrBot 插件（MCP 客户端，薄封装）
```

- **MCP 服务端**包含全部逻辑（状态/检测/映射/词库），通过 MCP 协议暴露工具
- **AstrBot 插件**是薄封装，在 `on_llm_request`/`on_llm_response` 中调用 MCP 服务

## 快速开始

### 1. 启动 MCP 服务端

```bash
cd random-persona-mcp
pip install fastmcp
python server.py
# → MCP server running on http://localhost:4568
```

### 2. 加载 AstrBot 插件

将 `astrbot_plugin_random_persona/` 放入 AstrBot `data/plugins/`，配置 MCP 地址。

## 命令

| 命令 | 功能 |
|------|------|
| `/persona` | 查看完整状态 |
| `/persona random` | 随机重置 |
| `/persona chill` / `warm` / `talkative` / `quiet` | 快速模式 |
| `/persona trait <维度> <0-1>` | 调节 OCEAN 特质 |
| `/persona emotion <标签>` | 手动触发情绪 |
| `/persona reset` | 重置人格+关系 |
| `/persona off` / `on` | 开关 |

## 词库

内嵌 200 词手标 PAD 种子（7 类情绪全覆盖）+ DUTIR 1,428 词扩展（jieba 词频过滤）。
总计 ~1,600 中文情感词，每个词标注 valence/arousal/dominance 连续值。

查询示例：
```
Mood(valence=0.7, arousal=0.5, dominance=0.6) → 倾向 "高兴、美好、开心、热情…"
Mood(valence=0.12, arousal=0.8, dominance=0.7) → 倾向 "恼怒、气愤、烦躁…"
```

## 对比

| | v1.0 | v2.1 |
|---|------|------|
| 状态层 | 1 (扁平4维) | 3 + Relationship |
| 情绪触发 | 无 | 11 种，规则引擎 |
| 话语层 | 5 种表达 | 17 种 Speech Act |
| 语言层 | 模糊描述 | 1600 词 PAD 词库 |
| prompt | ~180t | ~50t |
| 架构 | 单体插件 | MCP 服务 + 插件客户端 |

## 参考文献

OCC 评价理论 · PAD 情绪维度 · ALMA 三层架构 · EMA 情绪适应 · Gross 情绪调节 ·
社会渗透理论 · 面子协商理论 · Big Five 人格 · DUTIR 情感词汇本体 · NRC-VAD 词库

## License

MIT
