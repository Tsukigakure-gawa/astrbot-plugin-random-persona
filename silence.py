"""
Silence right: decide when the AI should respond minimally or not at all,
and generate appropriate short responses.

This is the "don't force a reply when there's nothing to say" mechanism
that makes the bot feel less mechanical.
"""

from __future__ import annotations

import random
import re
from typing import Any

# ---- silence responses per mode ----

_SILENCE: dict[str, list[str]] = {
    "记叙": ["嗯。", "知道了。", "行。", "好的。"],
    "说明": ["好。", "了解。", "收到。", "明白。"],
    "描写": ["嗯…看到了。", "好…", "哦。", "嗯。"],
    "议论": ["还行。", "就这样吧。", "没什么。", "一般。", "就这样。"],
    "抒情": ["👌", "好的~", "嗯嗯~", "好嘞。", "行吧~"],
}

# filler words injected with ~15% probability
_FILLERS: list[str] = [
    "嗯…", "啊", "嘛", "就是说", "反正",
    "话说", "说起来", "不过呢", "算了", "哎",
    "……", "唔", "啧",
]

# question indicator patterns
_QUESTION_RE = re.compile(
    r"[?？]|(吗|呢|吧|啥|什么|怎么|如何|为什么|谁|哪|"
    r"几|多少|能不能|可不可以|要不要|是不是|有没有)"
)


# ---- public API -----------------------------------------------------

class SilenceJudge:
    """Encapsulates silence-right logic and response generation."""

    @staticmethod
    def _has_question(text: str) -> bool:
        return bool(_QUESTION_RE.search(text or ""))

    def should_silence(self, state: dict[str, Any], user_message: str) -> bool:
        """Return True when the AI should respond minimally."""
        if not state.get("enabled", True):
            return False

        # A: user didn't ask anything + our last reply was substantial
        if (not self._has_question(user_message)
                and state.get("last_response_len", 0) > 150):
            return True

        # B: patience critically low
        if state["patience"] < 0.20:
            return True

        # C: low energy + cold affect → 30 % chance of silence
        if state["energy"] < 0.3 and state["valence"] < 0.3:
            if random.random() < 0.30:
                return True

        return False

    @staticmethod
    def get_silence_response(mode: str) -> str:
        responses = _SILENCE.get(mode, _SILENCE["说明"])
        return random.choice(responses)

    @staticmethod
    def maybe_add_filler(response: str, state: dict[str, Any]) -> str:
        """Inject a filler word/phrase at the end of *response* ~15 % of the time."""
        if random.random() < 0.15 and len(response) > 20:
            filler = random.choice(_FILLERS)
            stripped = response.rstrip()
            if not stripped.endswith(filler):
                return stripped + filler
        return response
