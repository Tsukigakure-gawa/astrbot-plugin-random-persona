"""
astrbot_plugin_random_persona
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

随机化 AI 情感基调和表达方式，避免永远积极 / 机械找话题。

核心机制
--------
- 四维情感状态 (energy / valence / patience / openness) 随对话自然漂移
- 五种表达方式 (记叙 / 说明 / 描写 / 议论 / 抒情) 加权随机切换
- 沉默权：判断「不需要回复」时发送极简回应而非硬找话题
- /persona 命令族让用户手动干预当前人格

Hooks
-----
on_llm_request (priority=-100)
    注入动态 persona prompt → 影响 LLM 写作风格

on_llm_response
    沉默权判断 · 状态更新 · 语气词注入
"""

from __future__ import annotations

import random
from typing import Any

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse

from .state import MODES, StateManager
from .prompt import PromptBuilder
from .silence import SilenceJudge


# ---- helpers --------------------------------------------------------

def _unwrap_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """AstrBot v4 wraps every config value as ``{"value": actual, ...}``."""
    if not config:
        return {}
    result: dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, dict) and "value" in v:
            result[k] = v["value"]
        else:
            result[k] = v
    return result


def _extract_text_len(response: LLMResponse) -> int:
    """Return total text length across all Plain components."""
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
    "随机化AI情感基调和表达方式，五种表达方式+情感维度+沉默权，让回复更有人味",
    "1.0.0",
)
class RandomPersonaPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = _unwrap_config(config)

        # data dir
        from astrbot.api.star import StarTools

        self.data_dir = StarTools.get_data_dir("astrbot_plugin_random_persona")

        # engines
        self._state_mgr = StateManager(self.data_dir)
        self._prompt = PromptBuilder()
        self._silence = SilenceJudge()

        # config
        self._silence_mode: str = str(self.config.get("silence_mode", "短回应"))

    # -- lifecycle ---------------------------------------------------

    async def initialize(self) -> None:
        logger.info(
            "[RandomPersona] v1.0 已加载  |  "
            f"沉默模式={self._silence_mode}"
        )

    async def terminate(self) -> None:
        self._state_mgr.save()
        logger.info("[RandomPersona] 状态已保存，插件卸载")

    # -- LLM hooks ---------------------------------------------------

    @filter.on_llm_request(priority=-100)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        """在每次 LLM 请求前注入动态 persona 指令。"""
        try:
            session_id = event.unified_msg_origin
            state = self._state_mgr.get_or_init(session_id)

            user_msg = getattr(event, "message_str", "") or ""
            self._state_mgr.drift(state, str(user_msg))

            block = self._prompt.build(state)
            if block:
                req.system_prompt += f"\n\n{block}"

        except Exception:
            # never let persona injection crash the bot
            logger.debug("[RandomPersona] on_llm_request 异常，跳过注入", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(
        self, event: AstrMessageEvent, response: LLMResponse
    ) -> None:
        """后处理: 沉默权 + 语气词注入 + 状态更新。"""
        try:
            session_id = event.unified_msg_origin
            state = self._state_mgr.get_state(session_id)
            if state is None:
                return

            user_msg = getattr(event, "message_str", "") or ""

            # -- silence right --
            if self._silence.should_silence(state, str(user_msg)):
                short = self._silence.get_silence_response(state["mode"])
                if self._silence_mode == "完全不回":
                    # clear the response entirely
                    if response.result_chain is not None:
                        response.result_chain.chain.clear()
                    self._state_mgr.update_after_response(state, 0)
                    return

                # replace with short response
                chain = MessageChain()
                chain.chain.append(Plain(short))
                response.result_chain = chain
                self._state_mgr.update_after_response(state, len(short))
                return

            # -- filler injection --
            if response.result_chain is not None:
                for i, comp in enumerate(response.result_chain.chain):
                    if hasattr(comp, "text") and comp.text:
                        comp.text = self._silence.maybe_add_filler(
                            str(comp.text), state
                        )
                        break  # only inject into first text component

            # -- update state --
            resp_len = _extract_text_len(response)
            self._state_mgr.update_after_response(state, resp_len)

        except Exception:
            logger.debug("[RandomPersona] on_llm_response 异常", exc_info=True)

    # -- /persona commands --------------------------------------------

    def _session_id(self, event: AstrMessageEvent) -> str:
        return event.unified_msg_origin

    @filter.command("persona")
    async def cmd_persona_status(self, event: AstrMessageEvent) -> Any:
        """查看当前人格状态: /persona"""
        state = self._state_mgr.get_state(self._session_id(event))
        if state is None or not state.get("enabled"):
            yield event.plain_result(
                "🎭 随机人格当前未启用。\n"
                "发送 /persona on 开启。"
            )
            return

        from .prompt import _bar, _energy_desc, _valence_desc, _patience_desc, _mode_desc

        msg = (
            f"🎭 当前人格状态\n"
            f"能量: {_bar(state['energy'])} {state['energy']:.2f}  {_energy_desc(state['energy'])}\n"
            f"情绪: {_bar(state['valence'])} {state['valence']:.2f}  {_valence_desc(state['valence'])}\n"
            f"耐心: {_bar(state['patience'])} {state['patience']:.2f}  {_patience_desc(state['patience'])}\n"
            f"表达: {state['mode']}  ({_mode_desc(state['mode'])})\n"
            f"\n"
            f"━━ 可用命令 ━━\n"
            f"/persona random  — 随机重置人格\n"
            f"/persona chill   — 低能偏冷模式\n"
            f"/persona warm   — 高能偏暖模式\n"
            f"/persona talkative — 话多模式\n"
            f"/persona quiet — 少说模式\n"
            f"/persona mode <记叙|说明|描写|议论|抒情> — 切换表达方式\n"
            f"/persona off    — 关闭随机人格"
        )
        yield event.plain_result(msg)

    @filter.command("persona random")
    async def cmd_persona_random(self, event: AstrMessageEvent) -> Any:
        """随机重置: /persona random"""
        state = self._state_mgr.reset(self._session_id(event))
        from .prompt import _mode_desc
        yield event.plain_result(
            f"🎲 人格已随机重置！\n"
            f"当前表达方式: {state['mode']} ({_mode_desc(state['mode'])})"
        )

    @filter.command("persona chill")
    async def cmd_persona_chill(self, event: AstrMessageEvent) -> Any:
        """低能偏冷: /persona chill"""
        self._state_mgr.set_override(
            self._session_id(event),
            energy=0.2, valence=0.2, patience=0.5, openness=0.3,
        )
        yield event.plain_result("🧊 已切换为 chill 模式 (低能偏冷)")

    @filter.command("persona warm")
    async def cmd_persona_warm(self, event: AstrMessageEvent) -> Any:
        """高能偏暖: /persona warm"""
        self._state_mgr.set_override(
            self._session_id(event),
            energy=0.8, valence=0.8, patience=0.85, openness=0.8,
        )
        yield event.plain_result("☀️ 已切换为 warm 模式 (高能偏暖)")

    @filter.command("persona talkative")
    async def cmd_persona_talkative(self, event: AstrMessageEvent) -> Any:
        """话多: /persona talkative"""
        self._state_mgr.set_override(
            self._session_id(event),
            openness=0.9, energy=0.7, patience=0.8,
        )
        yield event.plain_result("🗣️ 已切换为 talkative 模式 (高开放性)")

    @filter.command("persona quiet")
    async def cmd_persona_quiet(self, event: AstrMessageEvent) -> Any:
        """少说: /persona quiet"""
        self._state_mgr.set_override(
            self._session_id(event),
            openness=0.1, energy=0.25, patience=0.35,
        )
        yield event.plain_result("🤫 已切换为 quiet 模式 (少说为妙)")

    @filter.command("persona mode")
    async def cmd_persona_mode(self, event: AstrMessageEvent, mode: str = "") -> Any:
        """切换表达方式: /persona mode <记叙|说明|描写|议论|抒情>"""
        mode = (mode or "").strip()
        if mode not in MODES:
            yield event.plain_result(
                f"❌ 无效模式。可选: {', '.join(MODES)}"
            )
            return
        self._state_mgr.set_override(self._session_id(event), mode=mode)
        from .prompt import _mode_desc
        yield event.plain_result(f"✏️ 表达方式已切换为: {mode} ({_mode_desc(mode)})")

    @filter.command("persona off")
    async def cmd_persona_off(self, event: AstrMessageEvent) -> Any:
        """关闭: /persona off"""
        self._state_mgr.set_override(self._session_id(event), enabled=False)
        yield event.plain_result("🛑 随机人格已关闭，恢复默认回复风格。\n发送 /persona on 重新开启。")

    @filter.command("persona on")
    async def cmd_persona_on(self, event: AstrMessageEvent) -> Any:
        """开启: /persona on"""
        self._state_mgr.set_override(self._session_id(event), enabled=True)
        yield event.plain_result("✅ 随机人格已开启！")
