"""
Gerenciamento do estado global do programa.

- ProgramState: dataclass com gesto, confiança, velocidade e contagem
- StateManager: escrita periódica do JSON e acesso thread-safe
"""

import logging
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from config import STATE_FILE, STATE_UPDATE_INTERVAL
from jsonio import atomic_write_json

log = logging.getLogger(__name__)


@dataclass
class ProgramState:
    gesture:  str   = "REST"
    confidence: float = 1.0
    speed:    float = 0.0
    count:    int   = 0


class StateManager:
    """
    Mantém o estado atual e o escreve em disco periodicamente
    em background, thread-safe.
    """

    def __init__(self, path: str = STATE_FILE,
                 interval: float = STATE_UPDATE_INTERVAL):
        self._state    = ProgramState()
        self._lock     = threading.Lock()
        self._path     = Path(path)
        self._interval = interval
        self._thread: threading.Thread | None = None
        self._running  = threading.Event()

    # ── Leitura / escrita do estado ────────────────────────────────────────

    def update(self, gesture: str, confidence: float,
               speed: float, count: int) -> None:
        with self._lock:
            self._state = ProgramState(gesture, confidence, speed, count)

    def get(self) -> ProgramState:
        with self._lock:
            return ProgramState(**asdict(self._state))

    # ── Persistência ────────────────────────────────────────────────────────

    def write_now(self) -> None:
        with self._lock:
            state = asdict(self._state)
        atomic_write_json(str(self._path), state)

    # ── Thread de background ────────────────────────────────────────────────

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("StateManager iniciado → %s", self._path)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.write_now()
        log.info("StateManager encerrado")

    def _loop(self) -> None:
        while self._running.is_set():
            self.write_now()
            time.sleep(self._interval)
