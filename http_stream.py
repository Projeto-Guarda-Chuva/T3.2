"""
Cliente WebSocket para o servidor de câmeras.

O servidor expõe uma página HTML que abre um WebSocket em
    ws://<host>:<port>/ws/<camera_key>
e envia frames JPEG como mensagens binárias (uma por frame).
Mensagens de texto são erros em JSON, ex: {"message": "Câmera indisponível."}

Requer a lib `websocket-client` (pip install websocket-client) —
não confundir com `websockets` (assíncrona), API diferente.
"""

import json
import logging
import time

import cv2
import numpy as np
import websocket

log = logging.getLogger(__name__)

_CONNECT_TIMEOUT  = 5.0    # s — timeout de conexão/handshake
_RECV_TIMEOUT     = 5.0    # s — timeout de recv por frame
_RECONNECT_DELAY  = 1.0    # s — espera entre tentativas


class WebSocketVideoStream:
    """
    Conecta a um servidor de câmera via WebSocket e entrega frames
    como np.ndarray BGR.

    Uso:
        stream = WebSocketVideoStream("ws://172.18.4.56:443/ws/emeet")
        stream.connect()
        while True:
            frame = stream.read()          # None se não houver frame ainda
            if frame is not None:
                cv2.imshow("stream", frame)
        stream.disconnect()
    """

    def __init__(self, url: str, max_failures: int = 5):
        self._url           = url
        self._max_failures  = max_failures
        self._ws:  websocket.WebSocket | None = None
        self._failures: int = 0

    # ── Conexão ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        self.disconnect()
        try:
            self._ws = websocket.create_connection(
                self._url, timeout=_CONNECT_TIMEOUT,
            )
            self._ws.settimeout(_RECV_TIMEOUT)
        except (OSError, websocket.WebSocketException) as e:
            log.error("Não foi possível conectar a %s — %s", self._url, e)
            self._ws = None
            return False

        self._failures = 0
        log.info("Conectado a %s", self._url)
        return True

    def disconnect(self) -> None:
        if self._ws:
            try:
                self._ws.close()
            except (OSError, websocket.WebSocketException):
                pass
            self._ws = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    # ── Leitura de frame ─────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """
        Tenta ler e retornar um frame BGR.
        Retorna None se não houver frame completo ainda (timeout, erro do
        servidor, ou reconexão em andamento).
        """
        if not self.connected:
            if self._failures >= self._max_failures:
                return None
            if not self._reconnect():
                return None

        try:
            opcode, data = self._ws.recv_data()
        except websocket.WebSocketTimeoutException:
            return None
        except (OSError, websocket.WebSocketException) as e:
            log.warning("Conexão perdida: %s", e)
            self._failures += 1
            self.disconnect()
            return None

        if opcode == websocket.ABNF.OPCODE_TEXT:
            # Mensagem de erro em JSON vinda do servidor
            try:
                msg = json.loads(data)
                log.warning("Servidor reportou erro: %s", msg.get("message", data))
            except (json.JSONDecodeError, TypeError):
                log.warning("Mensagem de texto inesperada: %r", data)
            return None

        if opcode != websocket.ABNF.OPCODE_BINARY or not data:
            return None

        self._failures = 0
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            log.debug("Falha ao decodificar JPEG")
        return frame

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _reconnect(self) -> bool:
        log.info("Reconectando (tentativa %d)...", self._failures + 1)
        time.sleep(_RECONNECT_DELAY)
        return self.connect()
