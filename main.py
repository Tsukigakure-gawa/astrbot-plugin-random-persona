"""
astrbot_plugin_random_persona  v2.1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AstrBot plugin — thin MCP client that delegates all persona logic
to the random-persona MCP server.

Hooks:
  on_llm_request  → persona_inject MCP call → append to system_prompt
  on_llm_response → persona_post_process MCP call → silence/filler

Commands:
  /persona * → persona_command MCP call
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.api.star import StarTools
from astrbot.core.message.message_event_result import MessageChain


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


class _MCPClient:
    """Minimal MCP SSE client for calling tools."""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: str | None = None

    def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool via SSE endpoint."""
        url = f"{self.base_url}/messages/?session_id={self._session or 'default'}"
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
            "id": 1,
        }).encode("utf-8")

        req = Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                # Parse SSE or plain JSON response
                if body.startswith("data:"):
                    body = body.split("data:", 1)[1].strip()
                result = json.loads(body)
                # Extract result from MCP response envelope
                if "result" in result:
                    inner = result["result"]
                    if isinstance(inner, dict) and "content" in inner:
                        content = inner["content"]
                        if isinstance(content, list) and len(content) > 0:
                            return {"text": content[0].get("text", "")}
                    return inner if isinstance(inner, dict) else {"text": str(inner)}
                return {"text": body}
        except URLError as e:
            logger.warning(f"[Persona] MCP call failed ({tool}): {e}")
            return {"text": "", "error": str(e)}
        except Exception as e:
            logger.warning(f"[Persona] MCP error ({tool}): {e}")
            return {"text": "", "error": str(e)}


# ---- plugin ---------------------------------------------------------

@register(
    "astrbot_plugin_random_persona",
    "Tsukigakure",
    "随机人格 MCP 客户端 — 委托 persona MCP 服务处理六层情感引擎",
    "2.1.0",
)
class RandomPersonaPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = _unwrap_config(config)

        # MCP server address
        self._mcp_url: str = str(self.config.get("mcp_url", "http://127.0.0.1:4568"))
        self._silence_mode: str = str(self.config.get("silence_mode", "短回应"))
        self._mcp = _MCPClient(self._mcp_url)

    async def initialize(self) -> None:
        logger.info(
            f"[RandomPersona] v2.1 MCP客户端已加载  |  "
            f"服务端={self._mcp_url}  |  沉默模式={self._silence_mode}"
        )

    async def terminate(self) -> None:
        logger.info("[RandomPersona] 插件卸载")

    # -- helpers -----------------------------------------------------

    def _session_id(self, event: AstrMessageEvent) -> str:
        return event.unified_msg_origin

    def _user_id(self, event: AstrMessageEvent) -> str:
        return f"{event.get_platform_name()}:{event.get_sender_id()}"

    # -- LLM hooks ---------------------------------------------------

    @filter.on_llm_request(priority=-100)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        try:
            sid = self._session_id(event)
            uid = self._user_id(event)
            user_msg = str(getattr(event, "message_str", "") or "")

            result = self._mcp._call("persona_inject", {
                "session_id": sid,
                "user_id": uid,
                "user_message": user_msg,
            })

            block = result.get("text", "")
            if block:
                req.system_prompt += f"\n\n{block}"

        except Exception:
            logger.debug("[RandomPersona] on_llm_request 异常", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: Any) -> None:
        try:
            sid = self._session_id(event)

            # Extract response text
            resp_text = ""
            if hasattr(response, "result_chain") and response.result_chain:
                for comp in response.result_chain.chain:
                    if hasattr(comp, "text") and comp.text:
                        resp_text += str(comp.text)

            result = self._mcp._call("persona_post_process", {
                "session_id": sid,
                "response_text": resp_text,
                "silence_mode": self._silence_mode,
            })

            if result.get("was_silenced"):
                text = result.get("text", "")
                if not text and self._silence_mode == "完全不回":
                    if hasattr(response, "result_chain") and response.result_chain:
                        response.result_chain.chain.clear()
                    return
                chain = MessageChain()
                chain.chain.append(Plain(text))
                if hasattr(response, "result_chain"):
                    response.result_chain = chain
                return

            # Apply filler from post-process
            new_text = result.get("text", resp_text)
            if new_text != resp_text and hasattr(response, "result_chain") and response.result_chain:
                for comp in response.result_chain.chain:
                    if hasattr(comp, "text") and comp.text:
                        comp.text = new_text
                        break

        except Exception:
            logger.debug("[RandomPersona] on_llm_response 异常", exc_info=True)

    # -- /persona commands --------------------------------------------

    @filter.command("persona")
    async def cmd_persona(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_status", {
            "session_id": self._session_id(event),
            "user_id": self._user_id(event),
        })
        yield event.plain_result(result.get("text", "无法获取状态"))

    @filter.command("persona random")
    async def cmd_random(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "user_id": self._user_id(event),
            "command": "random",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona chill")
    async def cmd_chill(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "chill",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona warm")
    async def cmd_warm(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "warm",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona talkative")
    async def cmd_talkative(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "talkative",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona quiet")
    async def cmd_quiet(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "quiet",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona off")
    async def cmd_off(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "off",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona on")
    async def cmd_on(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "on",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona reset")
    async def cmd_reset(self, event: AstrMessageEvent) -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "user_id": self._user_id(event),
            "command": "reset",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona trait")
    async def cmd_trait(self, event: AstrMessageEvent, dim: str = "", val: str = "") -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "trait",
            "args": f"{dim} {val}",
        })
        yield event.plain_result(result.get("text", ""))

    @filter.command("persona emotion")
    async def cmd_emotion(self, event: AstrMessageEvent, label: str = "") -> Any:
        result = self._mcp._call("persona_command", {
            "session_id": self._session_id(event),
            "command": "emotion",
            "args": label,
        })
        yield event.plain_result(result.get("text", ""))
