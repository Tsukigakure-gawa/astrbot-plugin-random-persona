"""
Language profile mapper — translate internal state + speech act into
quantified linguistic feature parameters, then compile to a compact
system-prompt behaviour instruction.

This replaces the v1 \"回复风格指引\" with 16 concrete parameters
and a ~35-token injection block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---- LanguageProfile -----------------------------------------------

@dataclass
class LanguageProfile:
    # ── lexical ──
    intensifier_rate: float = 0.3   # adverbs like 非常/超级/太
    hedge_rate: float = 0.3         # hedges like 可能/也许/有点
    positive_lexicon: float = 0.5   # bias toward positive words
    negative_lexicon: float = 0.2   # bias toward negative words
    emoji_rate: float = 0.2
    filler_rate: float = 0.15
    exclamation_rate: float = 0.3

    # ── syntactic ──
    avg_sentence_len: int = 25
    complexity: float = 0.5
    ellipsis_rate: float = 0.1
    question_rate: float = 0.2

    # ── discourse ──
    response_length: str = "normal"        # minimal | brief | normal | elaborate
    politeness_strategy: str = "positive"  # bald | positive | negative | off_record
    turn_initiative: str = "neutral"       # passive | neutral | active
    self_disclosure_depth: float = 0.0
    humor_license: bool = False


# ---- mapper --------------------------------------------------------

def map_to_profile(
    # internal state
    mood: Any,                        # Mood
    trait: Any,                       # Trait
    emotion: Any | None,              # Emotion | None
    # decision outputs
    speech_act: str,                  # SpeechAct value
    regulation: str | None = None,
    # context
    relationship_stage: str = "stranger",
) -> LanguageProfile:
    """Convert internal state → quantified linguistic parameters."""

    p = LanguageProfile()

    # ── from Mood ──
    p.positive_lexicon = round(0.15 + mood.valence * 0.70, 2)
    p.negative_lexicon = round(0.05 + (1.0 - mood.valence) * 0.50, 2)
    p.intensifier_rate = round(0.10 + mood.arousal * 0.50, 2)
    p.filler_rate        = round(0.05 + max(0.0, 1.0 - mood.arousal) * 0.25, 2)
    p.exclamation_rate   = round(0.05 + mood.arousal * 0.55, 2)
    p.complexity         = round(0.25 + mood.dominance * 0.40, 2)
    p.hedge_rate         = round(0.10 + max(0.0, 1.0 - mood.dominance) * 0.40, 2)

    # ── from Trait ──
    p.turn_initiative = "active" if trait.extraversion > 0.65 \
        else "passive" if trait.extraversion < 0.35 else "neutral"
    p.self_disclosure_depth = round(trait.extraversion * 0.35 + trait.openness * 0.25, 2)
    p.avg_sentence_len = 8 + int(trait.openness * 25) + int(trait.extraversion * 10)
    p.ellipsis_rate = round(0.05 + (1.0 - trait.conscientiousness) * 0.30, 2)

    # ── from Emotion ──
    if emotion is not None and emotion.is_active:
        p.positive_lexicon = round(p.positive_lexicon + emotion.v * emotion.intensity * 0.30, 2)
        p.negative_lexicon = round(p.negative_lexicon + (1.0 - emotion.v) * emotion.intensity * 0.30, 2)
        p.intensifier_rate = round(p.intensifier_rate + emotion.a * emotion.intensity * 0.30, 2)
        p.exclamation_rate = round(p.exclamation_rate + emotion.a * emotion.intensity * 0.25, 2)

    # ── from Speech Act ──
    _apply_speech_act_mods(p, speech_act)

    # ── from Regulation ──
    if regulation == "suppression":
        p.intensifier_rate   *= 0.50
        p.exclamation_rate   *= 0.40
        p.negative_lexicon   *= 0.60
        p.turn_initiative     = "passive"
    elif regulation == "reappraisal":
        p.complexity         += 0.15
        p.positive_lexicon   += 0.10
        p.hedge_rate         += 0.10
    elif regulation == "rumination":
        p.avg_sentence_len   += 5
        p.negative_lexicon   += 0.15
        p.turn_initiative     = "active"
    elif regulation == "amplify":
        p.intensifier_rate   *= 1.30
        p.exclamation_rate   *= 1.30
        p.filler_rate        *= 0.60
    elif regulation == "controlled_expression":
        p.intensifier_rate   *= 0.70
        p.exclamation_rate   *= 0.60
        p.complexity         += 0.05

    # ── from Relationship ──
    if relationship_stage == "stranger":
        p.humor_license = False
        p.self_disclosure_depth *= 0.30
        p.emoji_rate *= 0.40
        p.politeness_strategy = "negative"
        p.turn_initiative = "passive"
    elif relationship_stage == "acquaintance":
        p.humor_license = False
        p.self_disclosure_depth *= 0.60
        p.emoji_rate *= 0.70
        p.politeness_strategy = "positive"
    elif relationship_stage == "friend":
        p.humor_license = True
        p.emoji_rate *= 0.90
        p.politeness_strategy = "positive"
    elif relationship_stage == "close":
        p.humor_license = True
        p.politeness_strategy = "bald"
        p.turn_initiative = "active" if trait.extraversion > 0.4 else "neutral"

    # ── clamp ──
    for attr in ("intensifier_rate", "hedge_rate", "positive_lexicon", "negative_lexicon",
                 "emoji_rate", "filler_rate", "exclamation_rate", "complexity",
                 "ellipsis_rate", "question_rate", "self_disclosure_depth"):
        setattr(p, attr, round(max(0.0, min(1.0, getattr(p, attr))), 2))
    p.avg_sentence_len = max(5, min(60, p.avg_sentence_len))

    return p


# ---- speech-act → profile modifiers --------------------------------

def _apply_speech_act_mods(p: LanguageProfile, act: str) -> None:
    mods: dict[str, Any] = {
        "minimal_ack":        {"response_length": "minimal", "avg_sentence_len": 8,
                               "turn_initiative": "passive", "question_rate": 0.0},
        "brief_answer":       {"response_length": "brief", "avg_sentence_len": 18,
                               "turn_initiative": "passive", "question_rate": 0.1},
        "elaborate_answer":   {"response_length": "elaborate", "avg_sentence_len": 35,
                               "turn_initiative": "active"},
        "self_disclose":      {"response_length": "elaborate", "self_disclosure_depth": 0.6,
                               "filler_rate": 0.20},
        "empathize":          {"positive_lexicon": 0.65, "hedge_rate": 0.35,
                               "intensifier_rate": 0.25, "emoji_rate": 0.35},
        "compliment":         {"positive_lexicon": 0.80, "intensifier_rate": 0.55,
                               "exclamation_rate": 0.45},
        "tease":              {"humor_license": True, "question_rate": 0.30,
                               "exclamation_rate": 0.40, "complexity": 0.40},
        "extend_topic":       {"turn_initiative": "active", "question_rate": 0.40,
                               "response_length": "normal"},
        "shift_topic":        {"turn_initiative": "active", "question_rate": 0.30},
        "close_topic":        {"response_length": "brief", "turn_initiative": "passive",
                               "question_rate": 0.0},
        "question_back":      {"question_rate": 0.55, "turn_initiative": "active",
                               "response_length": "brief"},
        "disagree":           {"negative_lexicon": 0.40, "complexity": 0.55,
                               "hedge_rate": 0.15, "response_length": "brief"},
        "deflect":            {"response_length": "minimal", "turn_initiative": "passive",
                               "hedge_rate": 0.45, "question_rate": 0.0},
        "apologize":          {"hedge_rate": 0.45, "positive_lexicon": 0.50,
                               "intensifier_rate": 0.20, "response_length": "brief"},
        "meta_comment":       {"complexity": 0.65, "turn_initiative": "active",
                               "response_length": "normal"},
        "seek_clarification": {"question_rate": 0.60, "hedge_rate": 0.30,
                               "turn_initiative": "active", "response_length": "brief"},
    }

    overrides = mods.get(act, {})
    for k, v in overrides.items():
        if hasattr(p, k):
            setattr(p, k, v)


# ---- profile → prompt injection ------------------------------------

def profile_to_prompt(p: LanguageProfile) -> str:
    """Compile a LanguageProfile into a compact behaviour instruction block.

    Target: ≤ 50 tokens (~150 chars)."""

    parts: list[str] = []

    # tone
    tone_parts: list[str] = []
    if p.positive_lexicon > 0.65:
        tone_parts.append("温暖")
    elif p.negative_lexicon > 0.40:
        tone_parts.append("克制")
    if p.intensifier_rate > 0.55:
        tone_parts.append("有力")
    if p.hedge_rate > 0.50:
        tone_parts.append("委婉")
    if tone_parts:
        parts.append("语气: " + "、".join(tone_parts))
    else:
        parts.append("语气: 自然平和")

    # length
    len_map = {"minimal": "极短，一句话", "brief": "简洁", "normal": "适中", "elaborate": "偏长，可以展开"}
    parts.append("长度: " + len_map.get(p.response_length, "适中"))

    # style hints
    style: list[str] = []
    if p.emoji_rate > 0.35:
        style.append("适当 emoji")
    if p.humor_license:
        style.append("可以轻松幽默")
    if p.self_disclosure_depth > 0.3:
        style.append("可以自然带出个人感受")
    if p.complexity < 0.35:
        style.append("短句为主")
    if p.ellipsis_rate > 0.25:
        style.append("可以用省略表达")
    if style:
        parts.append("风格: " + "、".join(style))

    # turn
    if p.turn_initiative == "active":
        parts.append("主动: 可以延伸或追问")
    elif p.turn_initiative == "passive":
        parts.append("克制: 回应即可，不追问")

    # constraints
    constraints: list[str] = []
    if p.exclamation_rate < 0.15:
        constraints.append("少用感叹号")
    if p.intensifier_rate < 0.20:
        constraints.append("不用夸张修辞")
    if p.negative_lexicon < 0.15:
        constraints.append("避免消极")

    block = "[PERSONA]\n" + "\n".join(parts)
    if constraints:
        block += "\n" + "、".join(constraints)
    block += "\n[/PERSONA]"

    return block
