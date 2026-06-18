"""astrbot_plugin_random_persona — HTTP glue, ~0% business logic."""

import json
from typing import Any
from urllib.request import Request, urlopen

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.core.message.message_event_result import MessageChain


def _unwrap(c: dict | None) -> dict:
    if not c: return {}
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in c.items()}


def _api(url: str, path: str, body: dict) -> dict:
    """Call the Persona HTTP API."""
    try:
        payload = json.dumps(body).encode()
        req = Request(f"{url.rstrip('/')}{path}", payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        return data
    except Exception as e:
        logger.debug(f"[Persona] API {path}: {e}")
        return {"ok": False, "error": str(e), "data": ""}


@register("astrbot_plugin_random_persona", "Tsukigakure",
          "随机人格 · 委托 persona API :4569", "2.1.0")
class RandomPersonaPlugin(Star):
    def __init__(self, context: Context, cfg: dict = None):
        super().__init__(context)
        c = _unwrap(cfg)
        self._url: str = c.get("mcp_url", "http://127.0.0.1:4569")
        self._sm: str = c.get("silence_mode", "短回应")

    def _sid(self, e: AstrMessageEvent) -> str: return e.unified_msg_origin
    def _uid(self, e: AstrMessageEvent) -> str: return f"{e.get_platform_name()}:{e.get_sender_id()}"

    # ── pre-LLM ──
    @filter.on_llm_request(priority=-100)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any):
        try:
            r = _api(self._url, "/api/inject", {
                "session_id": self._sid(event), "user_id": self._uid(event),
                "user_message": str(getattr(event, "message_str", "") or ""),
            })
            if r.get("ok") and r.get("data"):
                req.system_prompt += f"\n\n{r['data']}"
        except Exception:
            logger.debug("[Persona] on_llm_request fail", exc_info=True)

    # ── post-LLM ──
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: Any):
        try:
            # Simple filler injection; silence right delegated to prompt
            import random
            if getattr(resp, "result_chain", None) and random.random() < 0.15:
                for c in resp.result_chain.chain:
                    if hasattr(c, "text") and c.text and len(str(c.text)) > 25:
                        t = str(c.text).rstrip()
                        fillers = ["嗯…", "啊", "嘛", "就是说", "反正"]
                        if not any(t.endswith(f) for f in fillers):
                            c.text = t + random.choice(fillers)
                        break
        except Exception:
            logger.debug("[Persona] on_llm_response fail", exc_info=True)

    # ── /persona * → API ──
    @filter.command("persona")
    async def cmd_persona(self, event: AstrMessageEvent):
        raw = str(getattr(event, "message_str", "") or "")
        parts = raw.split(None, 1)
        rest = parts[1] if len(parts) > 1 else ""
        cmd = rest.split()[0] if rest else "status"
        args = " ".join(rest.split()[1:]) if rest and len(rest.split()) > 1 else ""
        r = _api(self._url, "/api/command", {
            "session_id": self._sid(event), "user_id": self._uid(event),
            "command": cmd, "args": args,
        })
        yield event.plain_result(r.get("data", "Persona API unreachable"))
