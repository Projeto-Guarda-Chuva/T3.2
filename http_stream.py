"""
Cliente MJPEG sobre HTTP puro (sem dependências além de socket + OpenCV).

Usa socket TCP diretamente para controle de timeout e reconexão.
"""

import logging
import socket
import time
from urllib.parse import urlparse

import cv2
import numpy as np

log = logging.getLogger(__name__)

_CONNECT_TIMEOUT   = 5.0    # s — timeout de conexão TCP
_RECV_TIMEOUT      = 5.0    # s — timeout de recv
_RECONNECT_DELAY   = 1.0    # s — espera entre tentativas
_MAX_BUFFER        = 10 * 1024 * 1024  # 10 MB — limpa buffer se acumular demais


class HTTPVideoStream:
    """
    Conecta a um servidor MJPEG e entrega frames como np.ndarray BGR.

    Uso:
        stream = HTTPVideoStream("http://192.168.1.10:5000")
        stream.connect()
        while True:
            frame = stream.read()          # None se não houver frame ainda
            if frame is not None:
                cv2.imshow("stream", frame)
        stream.disconnect()
    """

    def __init__(self, url: str, max_failures: int = 5):
        parsed        = urlparse(url)
        self._host    = parsed.hostname
        self._port    = parsed.port or 80
        self._path    = parsed.path or "/video_feed"
        self._max_failures  = max_failures
        self._sock:   socket.socket | None = None
        self._buffer: bytes = b""
        self._boundary: str = ""
        self._failures: int = 0

    # ── Conexão ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Abre conexão TCP e lê o cabeçalho HTTP inicial."""
        self.disconnect()
        try:
            self._sock = socket.create_connection(
                (self._host, self._port), timeout=_CONNECT_TIMEOUT,
            )
            self._sock.settimeout(_RECV_TIMEOUT)
        except OSError as e:
            log.error("Não foi possível conectar a %s:%d — %s",
                      self._host, self._port, e)
            self._sock = None
            return False

        request = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}\r\n"
            "Connection: keep-alive\r\n"
            "Accept: multipart/x-mixed-replace;boundary=boundary\r\n"
            "User-Agent: GestureDetector/2.0\r\n"
            "\r\n"
        )
        self._sock.sendall(request.encode())

        # Ler até o fim dos headers HTTP
        header_bytes = b""
        for _ in range(200):
            try:
                chunk = self._sock.recv(4096)
            except OSError:
                break
            header_bytes += chunk
            if b"\r\n\r\n" in header_bytes:
                break

        self._boundary = self._parse_boundary(header_bytes.decode("utf-8", errors="replace"))
        self._buffer   = b""
        self._failures = 0
        log.info("Conectado a %s:%d  boundary='%s'",
                 self._host, self._port, self._boundary)
        return True

    def disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._buffer = b""

    @property
    def connected(self) -> bool:
        return self._sock is not None

    # ── Leitura de frame ─────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """
        Tenta ler e retornar um frame BGR.
        Retorna None se não houver frame completo ainda.
        Reconecta automaticamente em caso de falha.
        """
        if not self.connected:
            if self._failures >= self._max_failures:
                return None
            if not self._reconnect():
                return None

        # Receber novos dados do socket
        try:
            chunk = self._sock.recv(65536)
        except socket.timeout:
            return None
        except OSError as e:
            log.warning("Conexão perdida: %s", e)
            self._failures += 1
            self.disconnect()
            return None

        if not chunk:
            log.warning("Servidor encerrou a conexão")
            self._failures += 1
            self.disconnect()
            return None

        self._buffer += chunk
        self._failures = 0

        # Evitar acúmulo de buffer
        if len(self._buffer) > _MAX_BUFFER:
            log.warning("Buffer cheio — descartando")
            self._buffer = b""
            return None

        return self._extract_jpeg()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _reconnect(self) -> bool:
        log.info("Reconectando (tentativa %d)...", self._failures + 1)
        time.sleep(_RECONNECT_DELAY)
        return self.connect()

    def _extract_jpeg(self) -> np.ndarray | None:
        """Extrai um JPEG completo do buffer, se disponível."""
        ct_marker = b"Content-Type: image/jpeg"
        ct_pos    = self._buffer.find(ct_marker)
        if ct_pos == -1:
            return None

        hdr_end = self._buffer.find(b"\r\n\r\n", ct_pos)
        if hdr_end == -1:
            return None
        jpeg_start = hdr_end + 4

        boundary_bytes = f"--{self._boundary}".encode()
        boundary_pos   = self._buffer.find(boundary_bytes, jpeg_start)
        if boundary_pos == -1:
            return None

        jpeg_end = boundary_pos
        while jpeg_end > jpeg_start and self._buffer[jpeg_end - 1:jpeg_end] in (b"\r", b"\n"):
            jpeg_end -= 1

        jpeg_data = self._buffer[jpeg_start:jpeg_end]
        self._buffer = self._buffer[boundary_pos:]  # descartar frame consumido

        if not jpeg_data:
            return None

        arr   = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            log.debug("Falha ao decodificar JPEG")
        return frame

    @staticmethod
    def _parse_boundary(header: str) -> str:
        for line in header.splitlines():
            if "boundary=" in line:
                boundary = line.split("boundary=", 1)[1].strip().strip('"')
                if boundary.startswith("--"):
                    boundary = boundary[2:]
                return boundary
        return "boundary"
