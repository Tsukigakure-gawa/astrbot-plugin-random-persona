"""
Build a persona prompt block from the current emotional state.

The block is injected into the LLM system prompt via on_llm_request,
making the LLM adopt a tone, length, and conversational style that
matches the current persona dimensions.
"""

from __future__ import annotations

from typing import Any


# ---- visual bar ----

def _bar(value: float) -> str:
    filled = max(0, min(10, int(value * 10)))
    return "█" * filled + "░" * (10 - filled)


# ---- dimension descriptors ----

def _energy_desc(v: float) -> str:
    if v < 0.25:
        return "极低能量，慵懒困倦"
    if v < 0.45:
        return "偏低能量，不太想动"
    if v < 0.65:
        return "中等能量，精力尚可"
    if v < 0.85:
        return "偏高能量，比较活跃"
    return "高能量，兴奋活跃"


def _valence_desc(v: float) -> str:
    if v < 0.25:
        return "冷淡疏离"
    if v < 0.45:
        return "偏冷，理性克制"
    if v < 0.65:
        return "中性平和"
    if v < 0.85:
        return "偏暖，亲切友好"
    return "热情亲密"


def _patience_desc(v: float) -> str:
    if v < 0.25:
        return "几乎没耐心，容易敷衍"
    if v < 0.45:
        return "耐心有限，不想多说"
    if v < 0.65:
        return "耐心尚可"
    if v < 0.85:
        return "比较有耐心，愿意展开"
    return "很有耐心，可以详细回复"


_MODE_DESC: dict[str, str] = {
    "记叙": "讲述经历、描述过程",
    "说明": "客观陈述，条理清楚",
    "描写": "注重细节，具象化呈现",
    "议论": "给出分析、判断和观点",
    "抒情": "表达情绪感受，有人情味",
}


def _mode_desc(mode: str) -> str:
    return _MODE_DESC.get(mode, "自然表达")


# ---- style guidance -------------------------------------------------

def _style_guidance(state: dict[str, Any]) -> str:
    energy: float = state["energy"]
    valence: float = state["valence"]
    patience: float = state["patience"]
    openness: float = state["openness"]
    mode: str = state["mode"]

    lines: list[str] = []

    # length
    if patience < 0.3:
        lines.append("- 回复简短，点到即止，一两句话即可")
    elif patience > 0.7:
        lines.append("- 可以详细展开，放心多说几句")

    # tone
    if valence < 0.3:
        lines.append("- 语气冷静平实，不带感情色彩，不过分热情")
    elif valence > 0.7:
        lines.append("- 语气亲切温暖，自然地表达友好")

    if energy < 0.3:
        lines.append("- 少用感叹号和亢奋表达，保持低调")
    elif energy > 0.7:
        lines.append("- 可以适当活跃，增加能量感和节奏")

    # openness
    if openness < 0.3:
        lines.append("- 回应即可，不主动延伸话题或追问")
    elif openness > 0.7:
        lines.append("- 可以自然地延伸话题，适当追问或分享关联内容")

    # mode-specific
    mode_hints: dict[str, str] = {
        "记叙": "- 以讲述和叙事的语气组织回复，像在说一段经历",
        "说明": "- 以客观说明的方式，注重条理清晰和事实准确",
        "描写": "- 注重细节和画面感，用具体可感的语言描述",
        "议论": "- 敢于表达观点和判断，给出自己的分析和评价",
        "抒情": "- 自然流露情绪，让回复更有人味和温度",
    }
    lines.append(mode_hints.get(mode, "- 自然表达即可"))

    return "\n".join(lines)


# ---- public API -----------------------------------------------------

class PromptBuilder:
    """Turn an emotional state dict into an injectable system-prompt block."""

    def build(self, state: dict[str, Any]) -> str:
        if not state.get("enabled", True):
            return ""

        energy = state["energy"]
        valence = state["valence"]
        patience = state["patience"]
        mode = state["mode"]

        return (
            f"[PERSONA_STATE]\n"
            f"当前情感基调:\n"
            f"  能量: {_bar(energy)} {energy:.2f} ({_energy_desc(energy)})\n"
            f"  情绪: {_bar(valence)} {valence:.2f} ({_valence_desc(valence)})\n"
            f"  耐心: {_bar(patience)} {patience:.2f} ({_patience_desc(patience)})\n"
            f"  表达: {mode} ({_mode_desc(mode)})\n"
            f"\n"
            f"回复风格指引:\n"
            f"{_style_guidance(state)}\n"
            f"[/PERSONA_STATE]"
        )
