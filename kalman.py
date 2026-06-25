"""
Filtro de Kalman DWPA (Discrete White-noise Acceleration) 2-D.

Estado por joint: [x, y, vx, vy, ax, ay]  (6 estados)
Medição:          [x, y]                   (2 observações)

Modelo cinemático:
    x(t+dt)  = x + vx·dt + ½·ax·dt²
    vx(t+dt) = vx + ax·dt
    ax(t+dt) = ax  +  ruído de processo

O ruído de processo (qAcc) permite que a aceleração mude livremente entre
frames, o que modela bem o movimento humano. rMeas baixo faz o filtro
confiar mais nas detecções do YOLO do que na predição interna.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import (
    KALMAN_Q_ACC, KALMAN_R_MEAS, KALMAN_P_INIT,
    KALMAN_DT_MAX, KALMAN_JOINTS, KP_CONF_THRESHOLD,
)


class KalmanJoint:
    """Filtro de Kalman DWPA 6-state para um único joint 2-D."""

    def __init__(self, q_acc: float = KALMAN_Q_ACC, r_meas: float = KALMAN_R_MEAS):
        self.q_acc  = q_acc   # ruído de processo na aceleração
        self.r_meas = r_meas  # variância da medição
        self.s: Optional[np.ndarray] = None   # estado [x, y, vx, vy, ax, ay]
        self.P: Optional[np.ndarray] = None   # covariância 6×6

    @property
    def initialized(self) -> bool:
        return self.s is not None

    def _init(self, x: float, y: float) -> None:
        self.s = np.array([x, y, 0., 0., 0., 0.], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * KALMAN_P_INIT

    # ── Matriz de transição F para DWPA ───────────────────────────────────
    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        dt2h = 0.5 * dt * dt
        F = np.eye(6, dtype=np.float64)
        # posição ← velocidade e aceleração
        F[0, 2] = dt;  F[0, 4] = dt2h
        F[1, 3] = dt;  F[1, 5] = dt2h
        # velocidade ← aceleração
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    # ── Matriz de ruído de processo Q (DWPA) ──────────────────────────────
    def _process_noise(self, dt: float) -> np.ndarray:
        dt2h = 0.5 * dt * dt
        # Vetor de entrada G (separado por eixo)
        Gx = np.array([dt2h, 0, dt, 0, 1, 0], dtype=np.float64)
        Gy = np.array([0, dt2h, 0, dt, 0, 1], dtype=np.float64)
        return self.q_acc * (np.outer(Gx, Gx) + np.outer(Gy, Gy))

    def predict(self, dt: float) -> None:
        """Propaga o estado e a covariância para o próximo frame."""
        if not self.initialized:
            return
        dt = min(dt, KALMAN_DT_MAX)
        F  = self._transition(dt)
        Q  = self._process_noise(dt)
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q

    def update(self, x: float, y: float) -> None:
        """Corrige o estado com a medição (x, y) do detector."""
        if not self.initialized:
            self._init(x, y)
            return

        # H = [[1,0,0,0,0,0], [0,1,0,0,0,0]]
        H = np.zeros((2, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.eye(2, dtype=np.float64) * self.r_meas

        # Ganho de Kalman
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)   # 6×2

        # Atualização do estado
        innovation = np.array([x - self.s[0], y - self.s[1]], dtype=np.float64)
        self.s = self.s + K @ innovation

        # Covariância — forma de Joseph para estabilidade numérica
        I_KH  = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # ── Acessores de estado filtrado ───────────────────────────────────────
    @property
    def pos(self) -> tuple[float, float]:
        return float(self.s[0]), float(self.s[1])

    @property
    def vel(self) -> tuple[float, float]:
        return float(self.s[2]), float(self.s[3])

    @property
    def acc(self) -> tuple[float, float]:
        return float(self.s[4]), float(self.s[5])


@dataclass
class KalmanPerson:
    """
    Conjunto de filtros Kalman para os 8 joints de uma pessoa:
    ombros, cotovelos, quadris e pulsos.
    """
    joints: dict = field(default_factory=lambda: {
        name: KalmanJoint() for name in KALMAN_JOINTS
    })

    def update(self, keypoints: np.ndarray, dt: float,
               conf_thr: float = KP_CONF_THRESHOLD) -> None:
        """
        Atualiza todos os joints com os keypoints detectados.
        keypoints: array (17, 3) com [x, y, conf] por keypoint COCO.
        Joints abaixo de conf_thr apenas predizem (não atualizam).
        """
        for name, kp_idx in KALMAN_JOINTS.items():
            kf = self.joints[name]
            kf.predict(dt)
            x, y, conf = keypoints[kp_idx]
            if conf >= conf_thr:
                kf.update(float(x), float(y))

    def filtered_pos(self, joint_name: str,
                     keypoints: np.ndarray) -> tuple[float, float, float]:
        """
        Retorna (x, y, conf) filtrado para o joint dado.
        Se o filtro não estiver inicializado, retorna a medição bruta.
        """
        kp_idx = KALMAN_JOINTS[joint_name]
        x_raw, y_raw, conf = keypoints[kp_idx]
        kf = self.joints[joint_name]
        if kf.initialized:
            x_f, y_f = kf.pos
            return x_f, y_f, float(conf)
        return float(x_raw), float(y_raw), float(conf)
