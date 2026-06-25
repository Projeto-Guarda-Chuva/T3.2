"""
Visualização — todo código de desenho OpenCV centralizado aqui.
"""

import cv2
import numpy as np

from config import KP, SKELETON, SPEED_BAR_W, SPEED_BAR_H
from detector import Detection
from gesture_analyzer import Gesture


# ── Cores ───────────────────────────────────────────────────────────────────

def gesture_color(gesture: Gesture) -> tuple[int, int, int]:
    return {
        Gesture.SUBIR:   (0, 220, 0),
        Gesture.DESCER:  (0, 60,  220),
        Gesture.REPOUSO: (180, 180, 180),
    }[gesture]


def _kp_color(idx: int) -> tuple[int, int, int]:
    if idx <= 4:  return (50,  220, 50)
    if idx <= 8:  return (220, 180, 30)
    if idx <= 10: return (0,   120, 255)
    if idx <= 12: return (180, 50,  180)
    return (80, 200, 200)


# ── Desenho de esqueleto ────────────────────────────────────────────────────

def draw_skeleton(img: np.ndarray, det: Detection,
                  conf_thr: float = 0.3) -> None:
    DRAWN_JOINTS = {
        KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER,
        KP.LEFT_ELBOW,    KP.RIGHT_ELBOW,
        KP.LEFT_WRIST,    KP.RIGHT_WRIST,
        KP.LEFT_HIP,      KP.RIGHT_HIP,
    }

    kps = det.kps
    for a, b in SKELETON:
        if a not in DRAWN_JOINTS or b not in DRAWN_JOINTS:
            continue
        if kps[a, 2] < conf_thr or kps[b, 2] < conf_thr:
            continue
        cv2.line(img,
                 (int(kps[a, 0]), int(kps[a, 1])),
                 (int(kps[b, 0]), int(kps[b, 1])),
                 (200, 200, 200), 2, cv2.LINE_AA)

    for k in DRAWN_JOINTS:
        if kps[k, 2] < conf_thr:
            continue
        pt = (int(kps[k, 0]), int(kps[k, 1]))
        cv2.circle(img, pt, 5, _kp_color(k), -1, cv2.LINE_AA)
        cv2.circle(img, pt, 5, (0, 0, 0),    1,  cv2.LINE_AA)


def draw_bbox(img: np.ndarray, det: Detection,
              gesture: Gesture, person_idx: int) -> None:
    x, y, w, h = det.bbox.astype(int)
    col   = gesture_color(gesture)
    label = f"P{person_idx} {gesture.value}"

    cv2.rectangle(img, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bg = (x, max(0, y - th - 6), tw + 6, th + 6)
    cv2.rectangle(img, (bg[0], bg[1]), (bg[0] + bg[2], bg[1] + bg[3]), col, -1)
    cv2.putText(img, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_ref_lines(img: np.ndarray, det: Detection,
                   shoulder_y: float | None, hip_y: float | None) -> None:
    """Desenha linhas horizontais de referência de ombros e quadris."""
    x0 = int(det.bbox[0])
    x1 = int(det.bbox[0] + det.bbox[2])
    if shoulder_y is not None:
        cv2.line(img, (x0, int(shoulder_y)), (x1, int(shoulder_y)),
                 (0, 200, 255), 1, cv2.LINE_AA)
    if hip_y is not None:
        cv2.line(img, (x0, int(hip_y)), (x1, int(hip_y)),
                 (255, 100, 0), 1, cv2.LINE_AA)


# ── HUD ─────────────────────────────────────────────────────────────────────

def draw_hud(img: np.ndarray, fps: float, frame_idx: int,
             state: str, confidence: float,
             speed: float = 0.0, person_count: int = 0,
             source_label: str = "") -> None:
    """Cabeçalho de informações: FPS, estado, contagem, barra de velocidade."""
    gesture = Gesture(state) if state != "REST" else Gesture.REPOUSO
    col     = gesture_color(gesture)

    cv2.putText(img, f"Frame: {frame_idx}  FPS: {int(fps)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 10), 1, cv2.LINE_AA)

    cv2.putText(img,
                f"Estado: {state}  conf: {int(confidence * 100)}%  pessoas: {person_count}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Barra de velocidade (só quando há gesto ativo)
    if state != "REST":
        bx, by = 10, 68
        cv2.rectangle(img, (bx, by), (bx + SPEED_BAR_W, by + SPEED_BAR_H),
                      (80, 80, 80), -1)
        fill = int(speed * SPEED_BAR_W)
        if fill > 0:
            cv2.rectangle(img, (bx, by), (bx + fill, by + SPEED_BAR_H), col, -1)
        cv2.rectangle(img, (bx, by), (bx + SPEED_BAR_W, by + SPEED_BAR_H),
                      (200, 200, 200), 1)
        cv2.putText(img, f"speed: {speed:.2f}",
                    (bx + SPEED_BAR_W + 8, by + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    if source_label:
        cv2.putText(img, source_label,
                    (img.shape[1] - 200, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_progress_bar(img: np.ndarray, pos: int, total: int) -> None:
    """Barra de progresso na base da janela (só para vídeo pré-gravado)."""
    if total <= 0:
        return
    bar_w  = img.shape[1] - 20
    filled = int(bar_w * pos / total)
    by     = img.shape[0] - 8
    cv2.rectangle(img, (10, by), (10 + bar_w, by + 6), (60, 60, 60), -1)
    cv2.rectangle(img, (10, by), (10 + filled, by + 6), (0, 200, 255), -1)
