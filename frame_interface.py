"""
Interface de Frames — Grupo 1
==============================
Captura frames do servidor de câmeras via WebSocket e os envia via named
pipe para o processo de inferência (Grupo 2).

O servidor expõe um WebSocket por câmera em ws://<host>:<port>/ws/<key>
(chaves conhecidas: "emeet", "fifine"), enviando cada frame como uma
mensagem binária JPEG.

Protocolo do pipe:
    [4 bytes: magic 0x47525544]
    [4 bytes: frame_id uint32]
    [4 bytes: width  uint32]
    [4 bytes: height uint32]
    [8 bytes: timestamp uint64 (µs)]
    [width × height × 3 bytes: BGR raw]
"""

import argparse
import logging
import os
import signal
import struct
import sys
import time
from urllib.parse import urlparse

import cv2
import numpy as np

# Adicionar o diretório do pacote ao path
sys.path.insert(0, os.path.dirname(__file__))

from config import HTTP_URL, PIPE_PATH, FPS_LIMIT
from http_stream import WebSocketVideoStream

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("frame_interface")

# ── Protocolo de frames ──────────────────────────────────────────────────────
FRAME_MAGIC        = 0x47525544
HEADER_FORMAT      = "<IIIII Q"       # magic, frame_id, w, h, channels, timestamp_us
HEADER_SIZE        = struct.calcsize(HEADER_FORMAT)
MAX_PIPE_BUFFER    = 10               # frames máximos no pipe antes de throttle
MAX_CONSEC_ERRORS  = 10


def build_ws_url(base_url: str, camera: str) -> str:
    """
    Monta a URL do WebSocket a partir da URL base do servidor (HTTP_URL,
    ex: http://172.18.4.56:443) e da chave da câmera (ex: 'emeet').
    Resulta em: ws://172.18.4.56:443/ws/emeet
    """
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path  # tolera URL sem esquema
    return f"{scheme}://{netloc}/ws/{camera}"


def encode_frame(frame: np.ndarray, frame_id: int) -> bytes:
    """Serializa um frame BGR em bytes para envio pelo pipe."""
    h, w = frame.shape[:2]
    ts   = int(time.monotonic_ns() // 1000)  # microsegundos
    header = struct.pack(HEADER_FORMAT,
                         FRAME_MAGIC, frame_id, w, h, 3, ts)
    return header + frame.tobytes()


def create_pipe(path: str) -> None:
    """Remove e recria o named pipe."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    os.mkfifo(path, 0o666)
    log.info("Pipe criado: %s", path)


# ── Loop principal ───────────────────────────────────────────────────────────

def run(ws_url: str, pipe_path: str = PIPE_PATH,
        fps_limit: int = FPS_LIMIT) -> int:
    running    = True
    frame_id   = 0
    errors     = 0
    frames_sent = 0
    t_start    = time.monotonic()
    dt_target  = 1.0 / fps_limit

    def _shutdown(sig, _frame):
        nonlocal running
        log.info("Encerrando (sinal %d)...", sig)
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    log.info("URL: %s", ws_url)
    log.info("Pipe: %s", pipe_path)

    # Criar pipe e aguardar o Grupo 2 abrir a leitura
    create_pipe(pipe_path)
    log.info("Aguardando Grupo 2 conectar ao pipe...")
    pipe_fd = os.open(pipe_path, os.O_WRONLY)  # bloqueia até o leitor conectar
    log.info("Grupo 2 conectado!")

    stream = WebSocketVideoStream(ws_url)
    if not stream.connect():
        log.error("Falha ao conectar ao servidor WebSocket")
        os.close(pipe_fd)
        return 1
    log.info("Stream WebSocket conectado")

    try:
        while running:
            t_loop = time.monotonic()

            frame = stream.read()

            if frame is None:
                errors += 1
                log.warning("Frame vazio (%d/%d)", errors, MAX_CONSEC_ERRORS)
                if errors >= MAX_CONSEC_ERRORS:
                    log.error("Muitos erros consecutivos — encerrando")
                    break
                time.sleep(0.1)
                continue

            errors    = 0
            frame_id += 1
            frames_sent += 1

            payload = encode_frame(frame, frame_id)
            try:
                os.write(pipe_fd, payload)
            except BrokenPipeError:
                log.warning("Pipe quebrado — tentando reconectar...")
                os.close(pipe_fd)
                try:
                    pipe_fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                except OSError as e:
                    log.error("Não foi possível reabrir pipe: %s", e)
                    break

            if frame_id % 100 == 0:
                elapsed = time.monotonic() - t_start
                log.info("Frame %d  %dx%d  FPS=%.1f",
                         frame_id, frame.shape[1], frame.shape[0],
                         frames_sent / elapsed)

            # Controle de FPS
            elapsed_loop = time.monotonic() - t_loop
            sleep_for    = dt_target - elapsed_loop
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        stream.disconnect()
        os.close(pipe_fd)
        try:
            os.unlink(pipe_path)
        except FileNotFoundError:
            pass

    elapsed = time.monotonic() - t_start
    log.info("Encerrado — %d frames em %.1f s (%.1f FPS)",
             frames_sent, elapsed, frames_sent / max(elapsed, 1e-6))
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interface de Frames — Grupo 1")
    p.add_argument("--url", default=None,
                   help="URL completa do WebSocket (ex: ws://172.18.4.56:443/ws/emeet). "
                        "Se omitido, é montada a partir de --base-url e --camera.")
    p.add_argument("--base-url", default=HTTP_URL,
                   help=f"URL base do servidor de câmeras (padrão: {HTTP_URL})")
    p.add_argument("--camera", default="emeet", choices=["emeet", "fifine"],
                   help="Câmera a usar quando --url não é informado (padrão: emeet)")
    p.add_argument("--pipe", default=PIPE_PATH,
                   help=f"Caminho do named pipe (padrão: {PIPE_PATH})")
    p.add_argument("--fps", type=int, default=FPS_LIMIT,
                   help=f"Limite de FPS (padrão: {FPS_LIMIT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    url = args.url or build_ws_url(args.base_url, args.camera)
    sys.exit(run(url, args.pipe, args.fps))
