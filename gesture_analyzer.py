"""
Classificação de gestos e cálculo do coeficiente de velocidade.

Gestos:
  SUBIR   — pulsos acima da linha dos ombros
  DESCER  — pulsos abaixo da linha dos quadris
  REPOUSO — nenhum dos anteriores

Coeficiente de velocidade [0, 1]:
  SUBIR : dist(pulso, linha_ombro) / (SPEED_NORM_FACTOR × largura_ombro)
  DESCER: dist(pulso, linha_quadril) / (SPEED_NORM_FACTOR × largura_quadril)
  A normalização é relativa à escala da pessoa → funciona em qualquer distância.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from config import KP, KP_CONF_THRESHOLD, SPEED_NORM_FACTOR, MAX_PERSONS
from kalman import KalmanPerson
from detector import Detection


class Gesture(Enum):
    REPOUSO = "REST"
    SUBIR   = "UP"
    DESCER  = "DOWN"


# ── Utilidades ──────────────────────────────────────────────────────────────

def _mean_y(kps: np.ndarray, indices: list[int],
            conf_thr: float = KP_CONF_THRESHOLD) -> Optional[float]:
    """Média do eixo Y dos keypoints com confiança suficiente."""
    ys = [kps[i, 1] for i in indices if kps[i, 2] >= conf_thr]
    return float(np.mean(ys)) if ys else None


def _visible_xs(kps: np.ndarray, indices: list[int],
                conf_thr: float = KP_CONF_THRESHOLD) -> list[float]:
    return [kps[i, 0] for i in indices if kps[i, 2] >= conf_thr]


# ── Classificador de gesto ──────────────────────────────────────────────────

class GestureAnalyzer:
    """Classifica o gesto de uma pessoa baseado nos keypoints."""

    def __init__(self, conf_thr: float = KP_CONF_THRESHOLD):
        self.conf_thr = conf_thr

    def classify(self, det: Detection) -> Gesture:
        kps = det.kps
        shoulder_y = _mean_y(kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER], self.conf_thr)
        hip_y      = _mean_y(kps, [KP.LEFT_HIP,      KP.RIGHT_HIP],      self.conf_thr)
        wrist_y    = _mean_y(kps, [KP.LEFT_WRIST,    KP.RIGHT_WRIST],    self.conf_thr)

        if wrist_y is None:
            return Gesture.REPOUSO
        if shoulder_y is not None and wrist_y < shoulder_y:
            return Gesture.SUBIR
        if hip_y is not None and wrist_y > hip_y:
            return Gesture.DESCER
        return Gesture.REPOUSO

    def ref_ys(self, det: Detection) -> tuple[Optional[float], Optional[float]]:
        """Retorna (shoulder_y, hip_y) para visualização das linhas de referência."""
        return (
            _mean_y(det.kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER], self.conf_thr),
            _mean_y(det.kps, [KP.LEFT_HIP,      KP.RIGHT_HIP],      self.conf_thr),
        )


# ── Coeficiente de velocidade ───────────────────────────────────────────────

def _reference_width(kps: np.ndarray, indices: list[int],
                     bbox_w: float,
                     fallback_fraction: float,
                     conf_thr: float) -> Optional[float]:
    """
    Largura entre dois pontos (ex: ombro esquerdo e direito).
    Fallback para fração do bbox se só um lado estiver visível.
    """
    xs = _visible_xs(kps, indices, conf_thr)
    if len(xs) >= 2:
        return max(1.0, abs(xs[-1] - xs[0]))
    if len(xs) == 1:
        return max(1.0, bbox_w * fallback_fraction)
    return None


def compute_speed_coeff(
    det:       Detection,
    kp_filter: KalmanPerson,
    gesture:   Gesture,
    conf_thr:  float = KP_CONF_THRESHOLD,
) -> float:
    """
    Coeficiente de velocidade [0, 1] para uma pessoa.
    Usa posições filtradas pelo Kalman quando disponíveis.
    """
    if gesture == Gesture.REPOUSO:
        return 0.0

    kps = det.kps.copy()

    # Substituir posições brutas pelas filtradas pelo Kalman
    for joint_name in ("L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP",
                       "L_WRIST",    "R_WRIST"):
        x_f, y_f, conf = kp_filter.filtered_pos(joint_name, kps)
        from config import KALMAN_JOINTS
        idx = KALMAN_JOINTS[joint_name]
        kps[idx, 0] = x_f
        kps[idx, 1] = y_f
        # conf mantida original (não filtrada)

    bbox_w = float(det.bbox[2])

    if gesture == Gesture.SUBIR:
        ref_width = _reference_width(
            kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER],
            bbox_w, fallback_fraction=0.4, conf_thr=conf_thr,
        )
        ref_y  = _mean_y(kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER], conf_thr)
        wrist_y = _mean_y(kps, [KP.LEFT_WRIST, KP.RIGHT_WRIST], conf_thr)

        if ref_width is None or ref_y is None or wrist_y is None:
            return 0.0
        dist = ref_y - wrist_y   # positivo quando pulso está acima dos ombros

    else:  # DESCER
        ref_width = _reference_width(
            kps, [KP.LEFT_HIP, KP.RIGHT_HIP],
            bbox_w, fallback_fraction=0.3, conf_thr=conf_thr,
        )
        ref_y   = _mean_y(kps, [KP.LEFT_HIP, KP.RIGHT_HIP], conf_thr)
        wrist_y = _mean_y(kps, [KP.LEFT_WRIST, KP.RIGHT_WRIST], conf_thr)

        if ref_width is None or ref_y is None or wrist_y is None:
            return 0.0
        dist = wrist_y - ref_y   # positivo quando pulso está abaixo dos quadris

    if dist <= 0.0:
        return 0.0

    return float(np.clip(dist / (SPEED_NORM_FACTOR * ref_width), 0.0, 1.0))


# ── Agregação para múltiplas pessoas ───────────────────────────────────────

def classify_all(
    dets:       list[Detection],
    analyzers:  list[GestureAnalyzer],
    kp_filters: list[KalmanPerson],
    dt:         float,
    conf_thr:   float = KP_CONF_THRESHOLD,
) -> tuple[list[Gesture], str, float, float, int]:
    """
    Classifica gestos de todas as pessoas, calcula estado majoritário,
    confiança, velocidade média e contagem.

    Returns: (gestures, majority_state, confidence, mean_speed, count)
    """
    # Ordenar da esquerda para direita (indexação estável)
    dets = sorted(dets, key=lambda d: d.bbox[0])
    n    = min(len(dets), MAX_PERSONS)

    gestures: list[Gesture] = []
    for i in range(n):
        kp_filters[i].update(dets[i].kps, dt, conf_thr)
        gestures.append(analyzers[i].classify(dets[i]))

    majority_state = _majority(gestures)
    confidence     = _confidence(gestures, majority_state)

    target = Gesture(majority_state) if majority_state != "REST" else Gesture.REPOUSO
    matching = [
        (dets[i], kp_filters[i], gestures[i])
        for i in range(n)
        if gestures[i] == target
    ]
    count = len(matching)

    if count > 0 and target != Gesture.REPOUSO:
        mean_speed = float(np.mean([
            compute_speed_coeff(det, kf, g, conf_thr)
            for det, kf, g in matching
        ]))
    else:
        mean_speed = 0.0

    return gestures[:n], majority_state, confidence, mean_speed, count


def _majority(gestures: list[Gesture]) -> str:
    if not gestures:
        return "REST"
    counts = {g: gestures.count(g) for g in Gesture}
    winner = max(counts, key=counts.get)
    return winner.value


def _confidence(gestures: list[Gesture], majority: str) -> float:
    if not gestures:
        return 1.0
    target = Gesture(majority)
    return sum(1 for g in gestures if g == target) / len(gestures)
