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
    CENTROIDS_FILE,
)
from detector import OnnxBackend, preprocess, postprocess
from gesture_analyzer import GestureAnalyzer, classify_all
from jsonio import atomic_write_json
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
        use_gpu: bool = False, use_tensorrt: bool = False,
        headless: bool = False) -> int:
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
    prev_bboxes: list = []   # bboxes do frame anterior por slot (rastreamento de identidade)

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

    if headless:
        log.info("Modo headless — sem janela de visualização")
    else:
        cv2.namedWindow(f"{WINDOW_NAME} [Standalone]", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{WINDOW_NAME} [Standalone]", WINDOW_W, WINDOW_H)

    frame_count       = 0
    frames_per_second = 0
    display_fps       = 0.0
    t_start           = time.monotonic()
    t_fps             = t_start
    t_last_frame      = t_start
    paused            = False

    log.info("=== PROCESSANDO ===%s",
             "" if headless else
             f"  (q/ESC para sair{'  ESPAÇO para pausar' if not is_webcam else ''})")

    try:
        while running:
            # ── Pausa (só para arquivo, requer janela) ───────────────────
            if paused and not headless:
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
            gestures, state, conf, speed, count, centroids, prev_bboxes, ordered_dets = classify_all(
                dets, analyzers, kp_filters, dt, KP_CONF_THRESHOLD,
                prev_bboxes=prev_bboxes,
            )
            state_mgr.update(state, conf, speed, count)

            # ── JSON de centroides ────────────────────────────────────────
            if centroids:
                avg_cx = round(sum(cx for cx, _ in centroids) / len(centroids), 1)
                avg_cy = round(sum(cy for _, cy in centroids) / len(centroids), 1)
            else:
                avg_cx = avg_cy = None

            centroids_data = {
                "frame": frame_count,
                "t_sec": round(frame_count / src_fps, 3),
                "persons": [
                    {"slot": i, "cx": round(cx, 1), "cy": round(cy, 1)}
                    for i, (cx, cy) in enumerate(centroids)
                ],
                "avg": {"cx": avg_cx, "cy": avg_cy},
            }
            atomic_write_json(CENTROIDS_FILE, centroids_data)

            # ── Visualização (pulada em modo headless) ───────────────────
            if not headless:
                vis = frame.copy()
                for i, det_vis in enumerate(ordered_dets):
                    sy, hy = analyzers[i].ref_ys(det_vis)
                    draw_skeleton(vis, det_vis)
                    draw_ref_lines(vis, det_vis, sy, hy)
                    draw_bbox(vis, det_vis, gestures[i], i + 1)

                if not is_webcam and total_frames > 0:
                    draw_progress_bar(vis, frame_count, total_frames)

                draw_hud(vis, display_fps, frame_count,
                         state, conf, speed, count,
                         source_label=src_label)

                cv2.imshow(f"{WINDOW_NAME} [Standalone]", vis)

            if frame_count % LOG_INTERVAL == 0:
                log.info("Frame %d  FPS=%.0f  %s  conf=%.0f%%  speed=%.2f  n=%d/%d",
                         frame_count, display_fps, state,
                         conf * 100, speed, count, len(ordered_dets))

            # ── Teclas (requer janela; em headless, só Ctrl+C encerra) ───
            if not headless:
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
        if not headless:
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
    p.add_argument("--headless",  action="store_true",
                   help="Sem janela OpenCV (necessário sem display/X11)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(args.model, args.source, args.gpu, args.tensorrt, args.headless))
