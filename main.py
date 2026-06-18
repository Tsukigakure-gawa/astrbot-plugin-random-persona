"""
astrbot_plugin_random_persona  v2.0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Six-layer cognitive-affective-social architecture:
  Relationship → Trait → Mood → Emotion → Speech Act → Language → Response

v2 improvements over v1:
  - Three-layer state (Trait / Mood / Emotion) replacing flat dict
  - Ornstein-Uhlenbeck mean-reverting mood drift
  - Appraisal-driven emotion triggering from user messages
  - Speech Act selection between emotion and language
  - Quantified LanguageProfile → compact ~35t prompt injection
  - Per-user relationship model (social penetration theory)
  - ~80% reduction in prompt injection token cost
"""

from __future__ import annotations

import re
from typing import Any

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.api.star import StarTools
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse

from .state import (
    StateManager, JOY, SADNESS, ANGER, FEAR, SURPRISE,
    DISGUST, TRUST, ANTICIPATION, GUILT, GRATITUDE, HURT,
    EMOTION_LABELS,
)
from .appraisal import AppraisalEngine
from .speech_act import SpeechAct, select_speech_act
from .prompt import PromptBuilder
from .relationship import RelationshipManager


# ---- helpers --------------------------------------------------------

def _unwrap_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    result: dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, dict) and "value" in v:
            result[k] = v["value"]
        else:
            result[k] = v
    return result


def _has_question(text: str) -> bool:
    return bool(re.search(
        r"[?？]|(吗|呢|吧|啥|什么|怎么|如何|为什么|谁|哪|"
        r"几|多少|能不能|可不可以|要不要|是不是|有没有)", text or ""
    ))


_SILENCE_RESPONSES: dict[str, list[str]] = {
    "anger":       ["嗯。", "知道了。", "行。"],
    "sadness":     ["嗯…", "好。", "…"],
    "joy":         ["好的~", "👌", "好嘞。"],
    "hurt":        ["…", "嗯。", "好的。"],
    None:          ["好。", "了解。", "嗯。"],
}


def _extract_text_len(response: LLMResponse) -> int:
    if response.result_chain is None:
        return 0
    total = 0
    for comp in response.result_chain.chain:
        if hasattr(comp, "text") and comp.text:
            total += len(str(comp.text))
    return total


# ---- plugin ---------------------------------------------------------

@register(
    "astrbot_plugin_random_persona",
    "Tsukigakure",
    "六层认知-情感-社交架构: Trait→Mood→Emotion→SpeechAct→Language→Response",
    "2.0.0",
)
class RandomPersonaPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = _unwrap_config(config)

        self.data_dir = StarTools.get_data_dir("astrbot_plugin_random_persona")

        # engines
        self._state_mgr = StateManager(self.data_dir)
        self._appraiser = AppraisalEngine()
        self._prompt_builder = PromptBuilder()
        self._rel_mgr = RelationshipManager(self.data_dir)

        # config
        self._silence_mode: str = str(self.config.get("silence_mode", "短回应"))

    # -- lifecycle ---------------------------------------------------

    async def initialize(self) -> None:
        n_states = len(self._state_mgr.states)
        n_rels = len(self._rel_mgr._rels)
        logger.info(
            f"[RandomPersona] v2.0 已加载  |  "
            f"{n_states} 会话状态, {n_rels} 用户关系  |  "
            f"沉默模式={self._silence_mode}"
        )

    async def terminate(self) -> None:
        self._state_mgr.save()
        self._rel_mgr.save()
        logger.info("[RandomPersona] 状态已保存，插件卸载")

    # -- helpers -----------------------------------------------------

    def _user_id(self, event: AstrMessageEvent) -> str:
        return f"{event.get_platform_name()}:{event.get_sender_id()}"

    def _session_id(self, event: AstrMessageEvent) -> str:
        return event.unified_msg_origin

    # -- LLM hooks ---------------------------------------------------

    @filter.on_llm_request(priority=-100)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        try:
            sid = self._session_id(event)
            uid = self._user_id(event)
            state = self._state_mgr.get_or_init(sid)
            user_msg = str(getattr(event, "message_str", "") or "")

            # 1. mood drift
            self._state_mgr.drift_mood(sid)

            # 2. appraisal → emotion
            emo_result = self._appraiser.evaluate_with_regulation(
                user_msg, state.trait, state.mood
            )
            if emo_result:
                self._state_mgr.trigger_emotion(
                    sid, emo_result["label"], emo_result["intensity"]
                )

            # 3. emotion decay
            self._state_mgr.decay_emotion(sid)

            # 4. relationship
            rel = self._rel_mgr.get(uid)
            self._rel_mgr.record_interaction(
                uid,
                sentiment=state.mood.valence - 0.5,
                user_msg=user_msg,
            )

            # 5. silence check
            has_q = _has_question(user_msg)
            last_len = getattr(state, "last_response_len", 0)
            should_silence = (
                (not has_q and last_len > 150)
                or (self._state_mgr.patience(sid) < 0.20)
            )
            # relationship modulation
            silence_mod = self._rel_mgr.silence_mod(uid)

            # 6. speech act
            em_label = state.emotion.primary if state.emotion else None
            reg = emo_result.get("regulation") if emo_result else None
            sa = select_speech_act(
                emotion_label=em_label,
                regulation=reg,
                silence=should_silence and silence_mod > 0.5,
                user_has_question=has_q,
                relationship_stage=rel.stage,
                trait_extraversion=state.trait.extraversion,
                trait_openness=state.trait.openness,
            )

            # 7. prompt injection
            block = self._prompt_builder.build_for_injection(
                state=state,
                speech_act=sa.value if isinstance(sa, SpeechAct) else sa,
                regulation=reg,
                relationship_stage=rel.stage,
                silence_muted=should_silence,
            )
            if block:
                req.system_prompt += f"\n\n{block}"

            # 8. update counter
            state.msg_count += 1
            state.last_active = __import__('time').time()

        except Exception:
            logger.debug("[RandomPersona] on_llm_request 异常", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        try:
            sid = self._session_id(event)
            uid = self._user_id(event)
            state = self._state_mgr.get_state(sid)
            if state is None:
                return

            user_msg = str(getattr(event, "message_str", "") or "")
            rel = self._rel_mgr.get(uid)

            # silence check
            has_q = _has_question(user_msg)
            should_silence = (
                (not has_q and getattr(state, "last_response_len", 0) > 150)
                or (self._state_mgr.patience(sid) < 0.20)
            )

            if should_silence:
                mode_label = state.emotion.primary if state.emotion else None
                short = _SILENCE_RESPONSES.get(mode_label, _SILENCE_RESPONSES[None])
                import random
                text = random.choice(short)

                if self._silence_mode == "完全不回":
                    if response.result_chain is not None:
                        response.result_chain.chain.clear()
                    state.last_response_len = 0
                    return

                chain = MessageChain()
                chain.chain.append(Plain(text))
                response.result_chain = chain
                state.last_response_len = len(text)
                self._state_mgr._maybe_save()
                return

            # filler injection (15%)
            import random as _random
            if response.result_chain is not None and _random.random() < 0.15:
                for comp in response.result_chain.chain:
                    if hasattr(comp, "text") and comp.text and len(str(comp.text)) > 25:
                        fillers = ["嗯…", "啊", "嘛", "就是说", "反正", "话说", "不过呢", "算了"]
                        filler = _random.choice(fillers)
                        t = str(comp.text).rstrip()
                        if not t.endswith(filler):
                            comp.text = t + filler
                        break

            # update
            resp_len = _extract_text_len(response)
            state.last_response_len = resp_len
            self._state_mgr._maybe_save()

        except Exception:
            logger.debug("[RandomPersona] on_llm_response 异常", exc_info=True)

    # -- /persona commands --------------------------------------------

    @filter.command("persona")
    async def cmd_persona_status(self, event: AstrMessageEvent) -> Any:
        sid = self._session_id(event)
        uid = self._user_id(event)
        state = self._state_mgr.get_state(sid)

        if state is None or not state.enabled:
            yield event.plain_result(
                "🎭 随机人格未启用。\n发送 /persona on 开启。"
            )
            return

        rel = self._rel_mgr.get(uid)
        patience = self._state_mgr.patience(sid)

        msg = self._prompt_builder.build_status(
            state=state,
            patience=patience,
            relationship_stage=rel.stage,
        )
        yield event.plain_result(msg)

    @filter.command("persona random")
    async def cmd_persona_random(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.reset(self._session_id(event))
        state = self._state_mgr.get_state(self._session_id(event))
        lines = ["🎲 人格已随机重置！", f"外向性: {state.trait.extraversion:.2f}"]
        yield event.plain_result("\n".join(lines))

    @filter.command("persona chill")
    async def cmd_persona_chill(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.set_trait(self._session_id(event),
            extraversion=0.2, neuroticism=0.2, agreeableness=0.25)
        self._state_mgr.set_mood(self._session_id(event),
            valence=0.3, arousal=0.2, dominance=0.4)
        yield event.plain_result("🧊 chill 模式 (低外向·低唤醒·偏冷)")

    @filter.command("persona warm")
    async def cmd_persona_warm(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.set_trait(self._session_id(event),
            extraversion=0.8, agreeableness=0.85, neuroticism=0.2)
        self._state_mgr.set_mood(self._session_id(event),
            valence=0.8, arousal=0.7, dominance=0.6)
        yield event.plain_result("☀️ warm 模式 (高外向·高宜人·偏暖)")

    @filter.command("persona talkative")
    async def cmd_persona_talkative(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.set_trait(self._session_id(event),
            extraversion=0.9, openness=0.85)
        self._state_mgr.set_mood(self._session_id(event),
            arousal=0.7, valence=0.65)
        yield event.plain_result("🗣️ talkative 模式 (高外向·高开放)")

    @filter.command("persona quiet")
    async def cmd_persona_quiet(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.set_trait(self._session_id(event),
            extraversion=0.1, openness=0.2, neuroticism=0.25)
        self._state_mgr.set_mood(self._session_id(event),
            arousal=0.15, valence=0.4, dominance=0.35)
        yield event.plain_result("🤫 quiet 模式 (低外向·低唤醒)")

    @filter.command("persona off")
    async def cmd_persona_off(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.set_enabled(self._session_id(event), False)
        yield event.plain_result("🛑 随机人格已关闭。\n发送 /persona on 重新开启。")

    @filter.command("persona on")
    async def cmd_persona_on(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.set_enabled(self._session_id(event), True)
        yield event.plain_result("✅ 随机人格已开启！")

    @filter.command("persona reset")
    async def cmd_persona_reset(self, event: AstrMessageEvent) -> Any:
        self._state_mgr.reset(self._session_id(event))
        self._rel_mgr.reset(self._user_id(event))
        yield event.plain_result("🔄 人格 + 关系已重置。")

    @filter.command("persona trait")
    async def cmd_persona_trait(self, event: AstrMessageEvent, dim: str = "", val: str = "") -> Any:
        """手动调节特质: /persona trait <维度> <0.0-1.0>"""
        valid = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
        if dim not in valid:
            yield event.plain_result(f"❌ 无效维度。可选: {', '.join(sorted(valid))}")
            return
        try:
            v = float(val)
            if not 0.0 <= v <= 1.0:
                raise ValueError
        except (ValueError, TypeError):
            yield event.plain_result("❌ 值需在 0.0-1.0 之间")
            return
        self._state_mgr.set_trait(self._session_id(event), **{dim: v})
        yield event.plain_result(f"✅ {dim} → {v:.2f}")

    @filter.command("persona emotion")
    async def cmd_persona_emotion(self, event: AstrMessageEvent, label: str = "") -> Any:
        """手动触发情绪: /persona emotion <joy|anger|sadness|fear|...>"""
        label = (label or "").strip().lower()
        if label not in EMOTION_LABELS:
            yield event.plain_result(f"❌ 未知情绪。可选: {', '.join(EMOTION_LABELS)}")
            return
        self._state_mgr.trigger_emotion(self._session_id(event), label, 0.7)
        yield event.plain_result(f"⚡ 已触发: {label}")
