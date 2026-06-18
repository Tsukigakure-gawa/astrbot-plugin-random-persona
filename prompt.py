"""
Prompt injection builder — translate internal state into a compact
system-prompt behaviour instruction block.

v2:  delegates to language.py for quantified mapping, outputs ~35-token
     behaviour instructions instead of verbose v1 state tables.

Also provides the v1-compatible status display (bar-based) for /persona
commands.
"""

from __future__ import annotations

from typing import Any

from .language import map_to_profile, profile_to_prompt


# ---- visual bar (for /persona command output) ----------------------

def _bar(value: float) -> str:
    filled = max(0, min(10, int(value * 10)))
    return "█" * filled + "░" * (10 - filled)


def _energy_desc(v: float) -> str:
    if v < 0.25:  return "极低能量，慵懒困倦"
    if v < 0.45:  return "偏低能量，不太想动"
    if v < 0.65:  return "中等能量，精力尚可"
    if v < 0.85:  return "偏高能量，比较活跃"
    return "高能量，兴奋活跃"


def _valence_desc(v: float) -> str:
    if v < 0.25:  return "冷淡疏离"
    if v < 0.45:  return "偏冷，理性克制"
    if v < 0.65:  return "中性平和"
    if v < 0.85:  return "偏暖，亲切友好"
    return "热情亲密"


def _patience_desc(v: float) -> str:
    if v < 0.25:  return "几乎没耐心，容易敷衍"
    if v < 0.45:  return "耐心有限，不想多说"
    if v < 0.65:  return "耐心尚可"
    if v < 0.85:  return "比较有耐心，愿意展开"
    return "很有耐心，可以详细回复"


_OCEAN_LABELS: dict[str, str] = {
    "openness":          "开放性",
    "conscientiousness": "尽责性",
    "extraversion":      "外向性",
    "agreeableness":     "宜人性",
    "neuroticism":       "神经质",
}

_OCEAN_HIGH: dict[str, str] = {
    "openness":          "好奇·审美",
    "conscientiousness": "自律·条理",
    "extraversion":      "社交·活跃",
    "agreeableness":     "合作·共情",
    "neuroticism":       "敏感·易波动",
}

_OCEAN_LOW: dict[str, str] = {
    "openness":          "务实·惯例",
    "conscientiousness": "随性·灵活",
    "extraversion":      "内向·安静",
    "agreeableness":     "独立·直言",
    "neuroticism":       "稳定·抗压",
}


# ---- public API -----------------------------------------------------

class PromptBuilder:
    """Build system-prompt injection and status displays."""

    def build_for_injection(
        self,
        state: Any,              # SessionState
        speech_act: str,
        regulation: str | None = None,
        relationship_stage: str = "stranger",
        silence_muted: bool = False,
    ) -> str:
        """Return a compact behaviour block for req.system_prompt."""

        if not state.enabled:
            return ""

        profile = map_to_profile(
            mood=state.mood,
            trait=state.trait,
            emotion=state.emotion,
            speech_act=speech_act,
            regulation=regulation,
            relationship_stage=relationship_stage,
        )

        # if speech act says minimal, override profile
        if silence_muted:
            profile.response_length = "minimal"
            profile.turn_initiative = "passive"

        return profile_to_prompt(profile)

    def build_status(
        self,
        state: Any,              # SessionState
        patience: float,
        relationship_stage: str = "stranger",
        regulation: str | None = None,
    ) -> str:
        """Build /persona status display (v1-compatible format)."""

        t = state.trait
        m = state.mood
        em = state.emotion

        # OCEAN traits
        lines = ["🎭 人格基线 (Trait)"]
        for key in ("extraversion", "agreeableness", "openness", "conscientiousness", "neuroticism"):
            val = getattr(t, key, 0.5)
            label = _OCEAN_LABELS[key]
            high = _OCEAN_HIGH[key]
            low = _OCEAN_LOW[key]
            desc = high if val > 0.5 else low
            lines.append(f"  {label}: {_bar(val)} {val:.2f}  {desc}")

        # mood
        lines.append("")
        lines.append("🌊 当前心境 (Mood)")
        lines.append(f"  愉悦: {_bar(m.valence)} {m.valence:.2f}  ({_valence_desc(m.valence)})")
        lines.append(f"  唤醒: {_bar(m.arousal)} {m.arousal:.2f}  ({_energy_desc(m.arousal)})")
        lines.append(f"  支配: {_bar(m.dominance)} {m.dominance:.2f}")

        # derived
        lines.append("")
        lines.append("📊 计算属性")
        lines.append(f"  耐心: {_bar(patience)} {patience:.2f}  ({_patience_desc(patience)})")

        # emotion
        if em is not None and em.is_active:
            lines.append("")
            lines.append(f"⚡ 活跃情绪: {em.primary}")
            lines.append(f"  强度: {_bar(em.intensity)} {em.intensity:.2f}")
            if regulation:
                lines.append(f"  调节: {regulation}")

        # relationship
        if relationship_stage != STAGE_STRANGER:
            lines.append("")
            lines.append(f"👥 关系: {relationship_stage}")

        lines.append("")
        lines.append("━━ 命令 ━━")
        lines.append("/persona trait <维度> <值>  — 调节特质")
        lines.append("/persona emotion <标签>     — 手动触发情绪")
        lines.append("/persona random|chill|warm|talkative|quiet")
        lines.append("/persona off|on|reset")

        return "\n".join(lines)


# re-use relationship constants
from .relationship import STAGE_STRANGER
