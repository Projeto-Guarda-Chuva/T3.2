"""
Detector Standalone — webcam ou arquivo de vídeo
=================================================
Versão autossuficiente: não usa pipe nem frame_interface.
Captura direto da câmera ou de um arquivo e processa no mesmo processo.

Uso:
    python gesture_standalone.py <modelo.onnx>                  # webcam 0
    python gesture_standalone.py <modelo.onnx> --source 1       # webcam 1
    python gesture_standalone.py <modelo.onnx> --source video.mp4
    python gesture_standalone.py <modelo.onnx> --gpu --tensorrt
"""

import argparse
import logging
import os
import signal
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CONF_THRESHOLD, NMS_IOU, MAX_PERSONS,
    LOG_INTERVAL, WINDOW_NAME, WINDOW_W, WINDOW_H, KP_CONF_THRESHOLD,
)
from detector import OnnxBackend, preprocess, postprocess
from gesture_analyzer import GestureAnalyzer, classify_all
from kalman import KalmanPerson
from state import StateManager
from visualizer import (
    draw_skeleton, draw_bbox, draw_ref_lines, draw_hud, draw_progress_bar,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("standalone")


# ── Abertura da fonte de vídeo ───────────────────────────────────────────────

def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    """
    Abre a fonte de vídeo.
    Retorna (cap, is_webcam).
    source: "" → webcam 0; dígito → webcam N; string → arquivo.
    """
    if source == "" or source.isdigit():
        idx = int(source) if source.isdigit() else 0
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)    # minimizar latência
        return cap, True
    else:
        return cv2.VideoCapture(source), False


# ── Loop principal ───────────────────────────────────────────────────────────

def run(model_path: str, source: str = "",
        use_gpu: bool = False, use_tensorrt: bool = False) -> int:
    running = True

    def _shutdown(sig, _frame):
        nonlocal running
        log.info("Encerrando (sinal %d)...", sig)
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Modelo: %s  GPU=%s  TRT=%s", model_path, use_gpu, use_tensorrt)

    # Componentes
    backend    = OnnxBackend(model_path, use_gpu=use_gpu, use_tensorrt=use_tensorrt)
    state_mgr  = StateManager()
    state_mgr.start()

    analyzers  = [GestureAnalyzer()  for _ in range(MAX_PERSONS)]
    kp_filters = [KalmanPerson()     for _ in range(MAX_PERSONS)]

    cap, is_webcam = open_source(source)
    if not cap.isOpened():
        log.error("Não foi possível abrir fonte: %r", source or "webcam 0")
        return 1

    src_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # -1 para webcam
    src_label    = "WEBCAM" if is_webcam else f"VIDEO: {os.path.basename(source)}"
    dt_target    = 1.0 / src_fps

    log.info("Fonte: %s  %.0fx%.0f @ %.0f fps",
             src_label,
             cap.get(cv2.CAP_PROP_FRAME_WIDTH),
             cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
             src_fps)

    cv2.namedWindow(f"{WINDOW_NAME} [Standalone]", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"{WINDOW_NAME} [Standalone]", WINDOW_W, WINDOW_H)

    frame_count       = 0
    frames_per_second = 0
    display_fps       = 0.0
    t_start           = time.monotonic()
    t_fps             = t_start
    t_last_frame      = t_start
    paused            = False

    log.info("=== PROCESSANDO ===  (q/ESC para sair%s)",
             "  ESPAÇO para pausar" if not is_webcam else "")

    try:
        while running:
            # ── Pausa (só para arquivo) ──────────────────────────────────
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key in (27, ord("q")):
                    running = False
                continue

            # ── Captura ──────────────────────────────────────────────────
            t_frame = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                if not is_webcam:
                    log.info("Fim do vídeo.")
                else:
                    log.warning("Falha ao capturar frame.")
                break

            # ── dt real ──────────────────────────────────────────────────
            t_now     = time.monotonic()
            dt        = min(t_now - t_last_frame, 0.5)
            t_last_frame = t_now

            # ── Inferência ────────────────────────────────────────────────
            prep = preprocess(frame, backend.net_h, backend.net_w)
            outs = backend.run(prep)
            dets = postprocess(outs, prep, frame.shape[0], frame.shape[1],
                               CONF_THRESHOLD, NMS_IOU)

            frame_count       += 1
            frames_per_second += 1

            # FPS
            elapsed_fps = t_now - t_fps
            if elapsed_fps >= 1.0:
                display_fps       = frames_per_second / elapsed_fps
                frames_per_second = 0
                t_fps             = t_now

            # ── Classificar gestos ────────────────────────────────────────
            gestures, state, conf, speed, count = classify_all(
                dets, analyzers, kp_filters, dt, KP_CONF_THRESHOLD,
            )
            state_mgr.update(state, conf, speed, count)

            # ── Visualização ──────────────────────────────────────────────
            vis = frame.copy()
            n   = min(len(dets), MAX_PERSONS)
            for i in range(n):
                sy, hy = analyzers[i].ref_ys(dets[i])
                draw_skeleton(vis, dets[i])
                draw_ref_lines(vis, dets[i], sy, hy)
                draw_bbox(vis, dets[i], gestures[i], i + 1)

            if not is_webcam and total_frames > 0:
                draw_progress_bar(vis, frame_count, total_frames)

            draw_hud(vis, display_fps, frame_count,
                     state, conf, speed, count,
                     source_label=src_label)

            cv2.imshow(f"{WINDOW_NAME} [Standalone]", vis)

            if frame_count % LOG_INTERVAL == 0:
                log.info("Frame %d  FPS=%.0f  %s  conf=%.0f%%  speed=%.2f  n=%d/%d",
                         frame_count, display_fps, state,
                         conf * 100, speed, count, n)

            # ── Teclas ───────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                running = False
                break
            if key == ord(" ") and not is_webcam:
                paused = True

            # ── Throttle para vídeo pré-gravado ──────────────────────────
            if not is_webcam:
                elapsed_loop = time.monotonic() - t_frame
                sleep_for    = dt_target - elapsed_loop
                if sleep_for > 0:
                    time.sleep(sleep_for)

    finally:
        state_mgr.stop()
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - t_start
    log.info("Encerrado — %d frames em %.1f s (%.1f FPS médio)",
             frame_count, elapsed, frame_count / max(elapsed, 1e-6))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture Detector Standalone")
    p.add_argument("model",  help="Caminho para o modelo .onnx")
    p.add_argument("--source",    default="",    metavar="SRC",
                   help="Fonte de vídeo: '' = webcam 0, número = webcam N, "
                        "caminho = arquivo (padrão: '')")
    p.add_argument("--gpu",       action="store_true", help="Usar CUDA EP")
    p.add_argument("--tensorrt",  action="store_true", help="Usar TensorRT EP")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(args.model, args.source, args.gpu, args.tensorrt))
