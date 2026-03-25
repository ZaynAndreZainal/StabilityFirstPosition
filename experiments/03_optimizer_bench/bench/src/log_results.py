from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


class WallClock:
    def __init__(self):
        self.t0 = time.time()

    def seconds(self) -> float:
        return time.time() - self.t0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


class ResultLogger:
    def __init__(self, out_dir: str, filename: str = "metrics.json"):
        self.out_dir = out_dir
        self.filename = filename
        ensure_dir(out_dir)

    @property
    def path(self) -> str:
        return os.path.join(self.out_dir, self.filename)

    def save(self, payload: Dict[str, Any]) -> None:
        write_json(self.path, payload)