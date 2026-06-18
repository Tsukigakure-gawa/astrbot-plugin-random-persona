"""
Per-session emotional state management with drift and persistence.

Each conversation session maintains a dynamic emotional state across four
dimensions (energy, valence, patience, openness) plus a dominant expression
mode.  The state drifts naturally over time and reacts to user input patterns.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Optional

# ---- constants ----

MODES: list[str] = ["记叙", "说明", "描写", "议论", "抒情"]

DEFAULT_STATE: dict[str, Any] = {
    "energy": 0.60,
    "valence": 0.40,
    "patience": 0.72,
    "openness": 0.50,
    "mode": "议论",
    "enabled": True,
    "msg_count": 0,
    "last_response_len": 0,
    "last_active": 0,
}

# interaction keyword sets – keep these small; they're cheap pattern matches
LAUGH_KW: tuple[str, ...] = ("哈哈", "笑死", "😂", "233", "www", "草", "乐", "hhh")
LONG_MSG_THRESHOLD: int = 200
SHORT_MSG_THRESHOLD: int = 5
SILENCE_RESET_SECS: int = 1800  # 30 min
SILENCE_RESET_PROB: float = 0.30

DRIFT_ENERGY: float = 0.05
DRIFT_VALENCE: float = 0.05
DRIFT_PATIENCE: float = 0.02
DRIFT_OPENNESS: float = 0.03
MODE_SWITCH_PROB: float = 0.10

SAVE_EVERY_N: int = 10


# ---- helpers ----

def _clamp(v: float) -> float:
    return round(max(0.0, min(1.0, v)), 2)


# ---- StateManager ----

class StateManager:
    """Load, persist, initialise, drift, and query per-session emotional state."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.data_file = os.path.join(data_dir, "states.json")
        self.states: dict[str, dict[str, Any]] = {}
        self._dirty: int = 0
        self._load()

    # -- persistence -------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.data_file):
            return
        try:
            with open(self.data_file, "r", encoding="utf-8") as fh:
                self.states = json.load(fh)
        except Exception:
            self.states = {}

    def save(self) -> None:
        try:
            with open(self.data_file, "w", encoding="utf-8") as fh:
                json.dump(self.states, fh, ensure_ascii=False, indent=2)
            self._dirty = 0
        except Exception:
            pass

    # -- lifecycle ---------------------------------------------------

    def get_or_init(self, session_id: str) -> dict[str, Any]:
        """Return existing state or create a fresh randomised one."""
        if session_id not in self.states:
            self.states[session_id] = self._random_init()
            return self.states[session_id]

        state = self.states[session_id]
        elapsed = time.time() - state.get("last_active", 0)

        # long silence → chance to re-randomise
        if elapsed > SILENCE_RESET_SECS and random.random() < SILENCE_RESET_PROB:
            self.states[session_id] = self._random_init()

        return self.states[session_id]

    def _random_init(self) -> dict[str, Any]:
        s = dict(DEFAULT_STATE)
        s["energy"] = round(random.uniform(0.15, 0.95), 2)
        s["valence"] = round(random.uniform(0.15, 0.95), 2)
        s["patience"] = round(random.uniform(0.20, 0.95), 2)
        s["openness"] = round(random.uniform(0.15, 0.95), 2)
        s["mode"] = random.choice(MODES)
        s["msg_count"] = 0
        s["last_response_len"] = 0
        s["last_active"] = time.time()
        return s

    # -- drift -------------------------------------------------------

    def drift(self, state: dict[str, Any], user_message: str = "") -> None:
        """Apply natural + interaction-driven drift to *state* (mutated in-place)."""

        # natural drift
        state["energy"] = _clamp(state["energy"] + random.uniform(-DRIFT_ENERGY, DRIFT_ENERGY))
        state["valence"] = _clamp(state["valence"] + random.uniform(-DRIFT_VALENCE, DRIFT_VALENCE))
        state["patience"] = _clamp(state["patience"] - DRIFT_PATIENCE)
        state["openness"] = _clamp(state["openness"] + random.uniform(-DRIFT_OPENNESS, DRIFT_OPENNESS))

        # interaction effects
        text = str(user_message or "")
        if "?" in text or "？" in text:
            state["patience"] = _clamp(state["patience"] - 0.05)
        if any(kw in text.lower() for kw in LAUGH_KW):
            state["valence"] = _clamp(state["valence"] + 0.05)
            state["energy"] = _clamp(state["energy"] + 0.03)
        if len(text) > LONG_MSG_THRESHOLD:
            state["energy"] = _clamp(state["energy"] + 0.03)
        if 0 < len(text) < SHORT_MSG_THRESHOLD:
            state["openness"] = _clamp(state["openness"] - 0.02)

        # mode switch
        if random.random() < MODE_SWITCH_PROB:
            state["mode"] = random.choice(MODES)

        state["msg_count"] = state.get("msg_count", 0) + 1
        state["last_active"] = time.time()

        self._dirty += 1
        if self._dirty >= SAVE_EVERY_N:
            self.save()

    # -- utilities ---------------------------------------------------

    def update_after_response(self, state: dict[str, Any], response_len: int) -> None:
        state["last_response_len"] = response_len

    def get_state(self, session_id: str) -> dict[str, Any] | None:
        return self.states.get(session_id)

    def set_override(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Manually set one or more dimensions, persist, and return the state."""
        state = self.get_or_init(session_id)
        for k, v in kwargs.items():
            if k in state:
                state[k] = v
        self.save()
        return state

    def reset(self, session_id: str) -> dict[str, Any]:
        self.states[session_id] = self._random_init()
        self.save()
        return self.states[session_id]
