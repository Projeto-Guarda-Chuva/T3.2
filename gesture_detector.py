"""
Detector de Gestos — Grupo 2
==============================
Lê frames do named pipe, executa inferência YOLO11n-pose via ONNX Runtime,
classifica gestos, calcula velocidade e escreve o estado em JSON.

Uso:
    python gesture_detector.py <modelo.onnx> [--gpu] [--tensorrt]
"""

import argparse
import logging
import os
import select
import signal
import struct
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    PIPE_PATH, CONF_THRESHOLD, NMS_IOU, MAX_PERSONS,
    FRAME_QUEUE_MAX_SIZE, LOG_INTERVAL, WINDOW_NAME, WINDOW_W, WINDOW_H,
    KP_CONF_THRESHOLD,
)
from detector import OnnxBackend, preprocess, postprocess
from gesture_analyzer import GestureAnalyzer, classify_all
from kalman import KalmanPerson
from state import StateManager
from visualizer import draw_skeleton, draw_bbox, draw_ref_lines, draw_hud

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("gesture_detector")

# Protocolo de pipe (deve coincidir com frame_interface.py)
FRAME_MAGIC   = 0x47525544
HEADER_FORMAT = "<IIIII Q"
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)
READ_TIMEOUT  = 1.0   # segundos


# ── Leitura de frames do pipe ────────────────────────────────────────────────

def _read_exact(fd: int, n: int, timeout: float = READ_TIMEOUT) -> bytes | None:
    """Lê exatamente n bytes do fd, com timeout."""
    buf = b""
    deadline = time.monotonic() + timeout
    while len(buf) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            return None
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_frame(fd: int) -> tuple[np.ndarray, int] | None:
    """
    Lê um frame completo do pipe.
    Retorna (frame_bgr, frame_id) ou None em caso de timeout/erro.
    """
    header_bytes = _read_exact(fd, HEADER_SIZE)
    if header_bytes is None:
        return None

    magic, frame_id, w, h, channels, ts = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != FRAME_MAGIC:
        log.warning("Magic inválido: 0x%08X", magic)
        return None

    data_size = w * h * channels
    if data_size == 0 or data_size > 50 * 1024 * 1024:
        log.warning("Tamanho de frame inválido: %d", data_size)
        return None

    raw = _read_exact(fd, data_size)
    if raw is None:
        return None

    frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, channels)
    return frame, frame_id


# ── Loop principal ───────────────────────────────────────────────────────────

def run(model_path: str, use_gpu: bool = False,
        use_tensorrt: bool = False) -> int:
    running = True

    def _shutdown(sig, _frame):
        nonlocal running
        log.info("Encerrando (sinal %d)...", sig)
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    log.info("Modelo: %s  GPU=%s  TRT=%s", model_path, use_gpu, use_tensorrt)
    log.info("Pipe: %s", PIPE_PATH)

    # Inicializar componentes
    backend    = OnnxBackend(model_path, use_gpu=use_gpu, use_tensorrt=use_tensorrt)
    state_mgr  = StateManager()
    state_mgr.start()

    analyzers  = [GestureAnalyzer()  for _ in range(MAX_PERSONS)]
    kp_filters = [KalmanPerson()     for _ in range(MAX_PERSONS)]
    queue:   deque[tuple[np.ndarray, int]] = deque(maxlen=FRAME_QUEUE_MAX_SIZE)

    # Abrir pipe (bloqueia até o Grupo 1 abrir o lado de escrita)
    log.info("Aguardando Grupo 1 (frame_interface)...")
    pipe_fd = os.open(PIPE_PATH, os.O_RDONLY)
    log.info("Grupo 1 conectado!")

    # Janela de visualização
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    frame_count       = 0
    frames_per_second = 0
    display_fps       = 0.0
    t_start           = time.monotonic()
    t_fps             = t_start
    t_last_frame      = t_start

    log.info("=== PROCESSAMENTO INICIADO ===")

    try:
        while running:
            # ── Processar frame da fila ──────────────────────────────────
            if queue:
                frame, fid = queue.popleft()

                t_now = time.monotonic()
                dt    = min(t_now - t_last_frame, 0.5)
                t_last_frame = t_now

                prep   = preprocess(frame, backend.net_h, backend.net_w)
                outs   = backend.run(prep)
                dets   = postprocess(outs, prep, frame.shape[0], frame.shape[1],
                                     CONF_THRESHOLD, NMS_IOU)

                frame_count       += 1
                frames_per_second += 1

                # FPS
                elapsed_fps = t_now - t_fps
                if elapsed_fps >= 1.0:
                    display_fps       = frames_per_second / elapsed_fps
                    frames_per_second = 0
                    t_fps             = t_now

                # Classificar gestos e atualizar estado
                gestures, state, conf, speed, count = classify_all(
                    dets, analyzers, kp_filters, dt, KP_CONF_THRESHOLD,
                )
                state_mgr.update(state, conf, speed, count)

                # Visualização
                vis = frame.copy()
                n   = min(len(dets), MAX_PERSONS)
                for i in range(n):
                    sy, hy = analyzers[i].ref_ys(dets[i])
                    draw_skeleton(vis, dets[i])
                    draw_ref_lines(vis, dets[i], sy, hy)
                    draw_bbox(vis, dets[i], gestures[i], i + 1)
                draw_hud(vis, display_fps, frame_count,
                         state, conf, speed, count)
                cv2.imshow(WINDOW_NAME, vis)

                if frame_count % LOG_INTERVAL == 0:
                    log.info("Frame %d  FPS=%.0f  %s  conf=%.0f%%  speed=%.2f  n=%d/%d",
                             frame_count, display_fps, state,
                             conf * 100, speed, count, n)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                running = False
                break

            # ── Ler próximo frame do pipe ────────────────────────────────
            result = read_frame(pipe_fd)
            if result is not None:
                queue.append(result)
            else:
                time.sleep(0.005)

    finally:
        state_mgr.stop()
        os.close(pipe_fd)
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - t_start
    log.info("Encerrado — %d frames em %.1f s (%.1f FPS médio)",
             frame_count, elapsed, frame_count / max(elapsed, 1e-6))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture Detector — Grupo 2")
    p.add_argument("model", help="Caminho para o modelo .onnx")
    p.add_argument("--gpu",       action="store_true", help="Usar CUDA EP")
    p.add_argument("--tensorrt",  action="store_true", help="Usar TensorRT EP")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(args.model, use_gpu=args.gpu, use_tensorrt=args.tensorrt))
