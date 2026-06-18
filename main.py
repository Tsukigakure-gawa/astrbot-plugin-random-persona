"""astrbot_plugin_random_persona — MCP glue, ~1% of the logic."""

import json
from typing import Any
from urllib.request import Request, urlopen

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.api.star.filter.command import GreedyStr
from astrbot.core.message.message_event_result import MessageChain


def _unwrap(c: dict | None) -> dict:
    if not c: return {}
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in c.items()}


def _mcp(url: str, tool: str, args: dict) -> dict:
    try:
        payload = json.dumps({"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":tool,"arguments":args},"id":1}).encode()
        with urlopen(Request(f"{url}/messages/", payload,
            headers={"Content-Type":"application/json","Accept":"text/event-stream"}),
            timeout=10) as r:
            body = r.read().decode()
            if body.startswith("data:"): body = body.split("data:",1)[1].strip()
            res = json.loads(body)
            inner = res.get("result",{})
            if isinstance(inner, dict) and "content" in inner:
                content = inner["content"]
                if isinstance(content, list) and content:
                    return {"text": content[0].get("text","")}
            return inner if isinstance(inner, dict) else {"text": str(inner)}
    except Exception as e:
        logger.debug(f"[Persona] MCP {tool}: {e}")
        return {"text": "", "error": str(e)}


@register("astrbot_plugin_random_persona", "Tsukigakure",
          "随机人格 MCP 胶水层 · 委托 localhost:4568", "2.1.0")
class RandomPersonaPlugin(Star):
    def __init__(self, ctx: Context, cfg: dict = None):
        super().__init__(ctx)
        c = _unwrap(cfg)
        self._url: str = c.get("mcp_url", "http://127.0.0.1:4568")
        self._sm: str = c.get("silence_mode", "短回应")

    def _sid(self, e: AstrMessageEvent) -> str: return e.unified_msg_origin
    def _uid(self, e: AstrMessageEvent) -> str: return f"{e.get_platform_name()}:{e.get_sender_id()}"

    # ── pre-LLM: inject persona prompt ──
    @filter.on_llm_request(priority=-100)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any):
        try:
            r = _mcp(self._url, "persona_inject", {
                "session_id": self._sid(event), "user_id": self._uid(event),
                "user_message": str(getattr(event, "message_str", "") or ""),
            })
            if r.get("text"): req.system_prompt += f"\n\n{r['text']}"
        except Exception:
            logger.debug("[Persona] on_llm_request fail", exc_info=True)

    # ── post-LLM: silence + filler ──
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: Any):
        try:
            sid = self._sid(event)
            text = ""
            if getattr(resp, "result_chain", None):
                for c in resp.result_chain.chain:
                    if hasattr(c, "text") and c.text: text += str(c.text)

            r = _mcp(self._url, "persona_post_process", {
                "session_id": sid, "response_text": text, "silence_mode": self._sm,
            })
            if r.get("was_silenced"):
                t = r.get("text", "")
                if not t and self._sm == "完全不回":
                    if getattr(resp, "result_chain", None): resp.result_chain.chain.clear()
                    return
                chain = MessageChain(); chain.chain.append(Plain(t))
                if getattr(resp, "result_chain", None): resp.result_chain = chain
                return
            nt = r.get("text", text)
            if nt != text and getattr(resp, "result_chain", None):
                for c in resp.result_chain.chain:
                    if hasattr(c, "text") and c.text: c.text = nt; break
        except Exception:
            logger.debug("[Persona] on_llm_response fail", exc_info=True)

    # ── /persona * commands → MCP ──
    @filter.command("persona")
    async def cmd_persona(self, event: AstrMessageEvent, rest: GreedyStr = ""):
        rest = rest.strip()
        cmd, args = ("status", "") if not rest else (rest.split()[0], " ".join(rest.split()[1:]))
        r = _mcp(self._url, "persona_command", {
            "session_id": self._sid(event), "user_id": self._uid(event),
            "command": cmd, "args": args,
        })
        yield event.plain_result(r.get("text", "MCP unreachable"))
