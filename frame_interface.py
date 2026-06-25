"""
Interface de Frames — Grupo 1
==============================
Captura frames do servidor HTTP MJPEG e os envia via named pipe
para o processo de inferência (Grupo 2).

Protocolo do pipe:
    [4 bytes: magic 0x47525544]
    [4 bytes: frame_id uint32]
    [4 bytes: width  uint32]
    [4 bytes: height uint32]
    [8 bytes: timestamp uint64 (µs)]
    [width × height × 3 bytes: BGR raw]
"""

import logging
import os
import signal
import struct
import sys
import time

import cv2
import numpy as np

# Adicionar o diretório do pacote ao path
sys.path.insert(0, os.path.dirname(__file__))

from config import HTTP_URL, PIPE_PATH, FPS_LIMIT
from http_stream import HTTPVideoStream

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

def run(http_url: str = HTTP_URL, pipe_path: str = PIPE_PATH,
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

    log.info("URL: %s", http_url)
    log.info("Pipe: %s", pipe_path)

    # Criar pipe e aguardar o Grupo 2 abrir a leitura
    create_pipe(pipe_path)
    log.info("Aguardando Grupo 2 conectar ao pipe...")
    pipe_fd = os.open(pipe_path, os.O_WRONLY)  # bloqueia até o leitor conectar
    log.info("Grupo 2 conectado!")

    stream = HTTPVideoStream(http_url)
    if not stream.connect():
        log.error("Falha ao conectar ao servidor HTTP")
        os.close(pipe_fd)
        return 1
    log.info("Stream HTTP conectado")

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


if __name__ == "__main__":
    sys.exit(run())
