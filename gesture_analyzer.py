"""
Classificação de gestos e cálculo do coeficiente de velocidade.

Gestos:
  SUBIR     — pulsos acima da linha dos ombros
  DESCER    — pulsos abaixo da linha dos quadris
  FECHAR    — antebraços cruzados em X (convergindo para o centro)
  MUDAR_LED — braços estendidos paralelos à frente
  REPOUSO   — nenhum dos anteriores

Ordem de prioridade na classificação (do mais específico para o mais geral):
  FECHAR > MUDAR_LED > SUBIR > DESCER > REPOUSO

Coeficiente de velocidade [0, 1]:
  SUBIR : dist(pulso, linha_ombro) / (SPEED_NORM_FACTOR × largura_ombro)
  DESCER: dist(pulso, linha_quadril) / (SPEED_NORM_FACTOR × largura_quadril)
  Outros gestos retornam speed = 0.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

from config import (
    KP, KP_CONF_THRESHOLD, SPEED_NORM_FACTOR, MAX_PERSONS,
    CLOSE_ANG_MIN, CLOSE_ANG_MAX, CLOSE_MAX_WRIST_DIST,
    LED_MAX_VERT_RATIO, LED_MIN_HORIZ_RATIO, LED_MAX_ELBOW_BEND,
    OPEN_MIN_HORIZ_RATIO, OPEN_MAX_VERT_RATIO, OPEN_MAX_ELBOW_BEND,
    UP_MIN_RATIO, DOWN_MIN_RATIO,
    KALMAN_JOINTS,
)
from kalman import KalmanPerson
from detector import Detection


class Gesture(Enum):
    REPOUSO   = "REST"
    SUBIR     = "UP"
    DESCER    = "DOWN"
    FECHAR    = "CLOSE"
    MUDAR_LED = "LED"
    OPEN      = "OPEN"


# ── Primitivas geométricas ──────────────────────────────────────────────────

def _kp(kps: np.ndarray, idx: int,
        conf_thr: float = KP_CONF_THRESHOLD) -> Optional[tuple[float, float]]:
    """Retorna (x, y) se a confiança for suficiente, senão None."""
    if kps[idx, 2] >= conf_thr:
        return float(kps[idx, 0]), float(kps[idx, 1])
    return None


def _mean_y(kps: np.ndarray, indices: list[int],
            conf_thr: float = KP_CONF_THRESHOLD) -> Optional[float]:
    ys = [kps[i, 1] for i in indices if kps[i, 2] >= conf_thr]
    return float(np.mean(ys)) if ys else None


def _visible_xs(kps: np.ndarray, indices: list[int],
                conf_thr: float = KP_CONF_THRESHOLD) -> list[float]:
    return [kps[i, 0] for i in indices if kps[i, 2] >= conf_thr]


def _vec_angle_deg(ax: float, ay: float, bx: float, by: float) -> float:
    """
    Ângulo em graus do vetor (a→b) em relação ao eixo horizontal.
    Positivo = aponta para cima (y decresce em coords de imagem).
    """
    dx, dy = bx - ax, by - ay       # dy negativo = acima na imagem
    return math.degrees(math.atan2(-dy, dx))   # convencional: cima = +


def _angle_between(ax, ay, bx, by, cx, cy) -> float:
    """
    Ângulo interno no vértice B formado pelos segmentos BA e BC, em graus.
    Útil para medir dobramento do cotovelo (shoulder→elbow→wrist).
    """
    ux, uy = ax - bx, ay - by
    vx, vy = cx - bx, cy - by
    norm_u = math.hypot(ux, uy)
    norm_v = math.hypot(vx, vy)
    if norm_u < 1e-6 or norm_v < 1e-6:
        return 180.0
    cos_a = (ux * vx + uy * vy) / (norm_u * norm_v)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def _shoulder_width(kps: np.ndarray, bbox_w: float,
                    conf_thr: float = KP_CONF_THRESHOLD) -> float:
    """Largura entre ombros, com fallback para 40 % do bbox."""
    xs = _visible_xs(kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER], conf_thr)
    if len(xs) >= 2:
        return max(1.0, abs(xs[1] - xs[0]))
    return max(1.0, bbox_w * 0.4)


def _pose_reliable(kps: np.ndarray, bbox_w: float,
                    conf_thr: float = KP_CONF_THRESHOLD,
                    min_frac: float = 0.20) -> bool:
    """
    Sinaliza quando a pessoa está de perfil/costas para a câmera: a
    projeção 2D da largura dos ombros colapsa para poucos pixels mesmo com
    alta confiança de detecção, e QUALQUER métrica normalizada por essa
    largura (todos os gestos exceto FECHAR) perde significado físico —
    pequeno ruído de pixel passa a produzir ratios enormes.

    Confirmado em dados reais (calibração 26/06/2026, frames 575-577 de
    testeB.mp4): pessoa de costas gerou wrist_hip_dy_norm de até 6.6×,
    disparando falso DESCER mesmo com o limiar corrigido.
    """
    ls = _kp(kps, KP.LEFT_SHOULDER, conf_thr)
    rs = _kp(kps, KP.RIGHT_SHOULDER, conf_thr)
    if not (ls and rs):
        return False
    raw = abs(rs[0] - ls[0])
    return raw >= (bbox_w * min_frac)


# ── Detecção de cada gesto ──────────────────────────────────────────────────

def _is_fechar(kps: np.ndarray, bbox_w: float,
               conf_thr: float,
               ang_min: float    = CLOSE_ANG_MIN,
               ang_max: float    = CLOSE_ANG_MAX,
               max_wrist: float  = CLOSE_MAX_WRIST_DIST) -> bool:
    """
    Gesto FECHAR — antebraços cruzados em X.

    NOTA HISTÓRICA: a versão original exigia "cruzamento" estrito de sinal
    (dx do antebraço esquerdo positivo, direito negativo). Validação contra
    24 exemplos reais rotulados (calibração 26/06/2026) mostrou que ESSA
    CONDIÇÃO FALHA EM 100% DOS CASOS REAIS — na prática os pulsos ficam
    próximos ao centro sem de fato cruzar um sobre o outro. A lógica abaixo
    foi recalibrada a partir desses mesmos 24 exemplos.

    Estratégia nova: cada antebraço (cotovelo → pulso) deve apontar para
    cima/diagonal-cima (ângulo positivo, não muito próximo de 0°/horizontal
    nem de 180°/braço para o outro lado), E os pulsos devem estar próximos
    um do outro (não abertos lateralmente como em OPEN, nem como braços
    relaxados para baixo como em REST).

    A combinação ângulo-positivo + pulsos-próximos é o que discrimina:
      - UP também levanta os antebraços (ângulo parecido), mas os PULSOS
        FICAM AFASTADOS (cada braço sobe pelo seu próprio lado) —
        wrist_dist_norm > ~1.19 em todos os exemplos reais de UP.
      - LED/REST com pulsos por acaso próximos têm ângulo negativo (braço
        para baixo) ou próximo de 180° (braço quase horizontal) — fora da
        faixa [ang_min, ang_max].

    Calibração (24 exemplos reais de CLOSE vs 92 de outros gestos):
      left_forearm_angle  ∈ [86°, 143°]
      right_forearm_angle ∈ [28°, 117°]
      wrist_dist_norm     ∈ [0.04, 1.14]   (menor caso de UP real: 1.19)
    ang_min=20°/ang_max=150° e max_wrist=1.15 dão separação perfeita na
    amostra, mas a margem é estreita (gap real de só ~0.05 em wrist_dist
    entre o pior CLOSE e o melhor UP) — vale revalidar com mais dados.
    """
    le = _kp(kps, KP.LEFT_ELBOW,  conf_thr)
    lw = _kp(kps, KP.LEFT_WRIST,  conf_thr)
    re = _kp(kps, KP.RIGHT_ELBOW, conf_thr)
    rw = _kp(kps, KP.RIGHT_WRIST, conf_thr)

    if None in (le, lw, re, rw):
        return False

    # Ângulo de cada antebraço (90° = reto para cima; ver _vec_angle_deg)
    left_angle  = _vec_angle_deg(le[0], le[1], lw[0], lw[1])
    right_angle = _vec_angle_deg(re[0], re[1], rw[0], rw[1])

    if not (ang_min <= left_angle <= ang_max):
        return False
    if not (ang_min <= right_angle <= ang_max):
        return False

    # Pulsos acima dos cotovelos (em coordenadas de imagem: y_pulso < y_cotovelo).
    # No X real os antebraços apontam para cima, então os pulsos ficam
    # acima dos cotovelos. Em repouso/REST os pulsos caem abaixo ou ficam
    # no mesmo nível dos cotovelos — essa condição elimina esses falsos positivos.
    if lw[1] >= le[1]:   # pulso esquerdo não está acima do cotovelo esquerdo
        return False
    if rw[1] >= re[1]:   # pulso direito não está acima do cotovelo direito
        return False

    # Pulsos próximos (convergindo ao centro, não abertos como em OPEN)
    sw = _shoulder_width(kps, bbox_w, conf_thr)
    wrist_dist = math.hypot(lw[0] - rw[0], lw[1] - rw[1])
    if wrist_dist > max_wrist * sw:
        return False

    return True


def _is_mudar_led(kps: np.ndarray, bbox_w: float,
                  conf_thr: float,
                  max_vert:  float = LED_MAX_VERT_RATIO,
                  min_horiz: float = LED_MIN_HORIZ_RATIO,
                  max_bend:  float = LED_MAX_ELBOW_BEND) -> bool:
    """
    Gesto MUDAR_LED — braços estendidos paralelos à frente.

    Câmera posicionada acima e à frente: os braços estendidos aparecem
    como segmentos quase horizontais com pouca extensão (profundidade
    comprime o comprimento do braço).

    Condições (todas devem ser verdadeiras para ambos os lados):
      1. Pulso aproximadamente na mesma altura do ombro (± max_vert × sw).
      2. Pulso além do ombro lateralmente (ao menos min_horiz × sw para fora).
      3. Cotovelo razoavelmente estendido (ângulo shoulder→elbow→wrist
         próximo de 180° → bend = 180° - ângulo ≤ max_bend).

    CORREÇÃO (26/06/2026): a condição horizontal estava com o sinal
    invertido. Keypoints COCO são anatômicos (LEFT_SHOULDER = ombro
    esquerdo DA PESSOA), e para uma pessoa de FRENTE para a câmera — o
    caso normal — o ombro esquerdo dela aparece do lado DIREITO da
    imagem (espelhamento natural de estar de frente a alguém). A versão
    antiga exigia "pulso esquerdo mais à esquerda que o ombro esquerdo",
    o que só é fisicamente possível se a pessoa estiver de costas para a
    câmera. Confirmado em dados reais: TODOS os 19 exemplos rotulados de
    LED tinham as 3 pessoas de frente, com o sinal antigo sempre falhando.
    """
    ls = _kp(kps, KP.LEFT_SHOULDER,  conf_thr)
    le = _kp(kps, KP.LEFT_ELBOW,     conf_thr)
    lw = _kp(kps, KP.LEFT_WRIST,     conf_thr)
    rs = _kp(kps, KP.RIGHT_SHOULDER, conf_thr)
    re = _kp(kps, KP.RIGHT_ELBOW,    conf_thr)
    rw = _kp(kps, KP.RIGHT_WRIST,    conf_thr)

    if None in (ls, le, lw, rs, re, rw):
        return False

    sw = _shoulder_width(kps, bbox_w, conf_thr)

    # ── Lado esquerdo ───────────────────────────────────────────────────────
    left_vert_ok  = abs(lw[1] - ls[1]) <= max_vert  * sw
    # Pulso esquerdo deve se afastar lateralmente do ombro esquerdo, para o
    # lado DIREITO da imagem (pessoa de frente para a câmera)
    left_horiz_ok = (lw[0] - ls[0]) >= min_horiz * sw
    left_bend     = 180.0 - _angle_between(ls[0], ls[1],
                                           le[0], le[1],
                                           lw[0], lw[1])
    left_ext_ok   = left_bend <= max_bend

    # ── Lado direito ────────────────────────────────────────────────────────
    right_vert_ok  = abs(rw[1] - rs[1]) <= max_vert  * sw
    # Pulso direito deve se afastar lateralmente do ombro direito, para o
    # lado ESQUERDO da imagem (pessoa de frente para a câmera)
    right_horiz_ok = (rs[0] - rw[0]) >= min_horiz * sw
    right_bend     = 180.0 - _angle_between(rs[0], rs[1],
                                            re[0], re[1],
                                            rw[0], rw[1])
    right_ext_ok   = right_bend <= max_bend

    return (left_vert_ok  and left_horiz_ok  and left_ext_ok and
            right_vert_ok and right_horiz_ok and right_ext_ok)




def _is_open(kps: np.ndarray, bbox_w: float,
             conf_thr: float,
             min_horiz: float = OPEN_MIN_HORIZ_RATIO,
             max_vert:  float = OPEN_MAX_VERT_RATIO,
             max_bend:  float = OPEN_MAX_ELBOW_BEND) -> bool:
    """
    Gesto OPEN — T-pose: braços estendidos horizontalmente para os lados.

    Discriminação de MUDAR_LED: a extensão lateral é muito maior porque
    os braços estão no plano da câmera, não apontando para ela.

    Condições:
      1. O braço MAIS estendido deve passar de min_horiz × shoulder_width
         além do seu ombro (avaliado pelo MÁXIMO dos dois lados — tolera
         oclusão parcial assimétrica, quando um ator fica parcialmente na
         frente do outro).
      2. O braço MENOS estendido deve estar minimamente positivo (≥ 0.1 ×
         shoulder_width) — descarta casos onde um braço aponta para o lado
         oposto (como certos frames de LED com horiz negativo num lado).
      3. Pulso na mesma altura do ombro (avaliado pelo MÁXIMO dos dois
         lados, com margem mais generosa que o código anterior).
      4. Cotovelo razoavelmente estendido (MÁXIMO dos dois lados — tolera
         um braço parcialmente dobrado por oclusão, como confirmado em
         frames 205-235 com P0 parcialmente ocluído por P1).

    CORREÇÕES (26/06/2026):
    a) Sinal horizontal invertido — ver docstring _is_mudar_led.
    b) Lógica de max/min em vez de "ambos individualmente": calibrada com
       31 exemplos reais de OPEN. OPEN_MIN_HORIZ_RATIO = 1.0 continua
       sendo o threshold numérico certo — só mudou como é avaliado.

    Calibração: 31/31 exemplos rotulados de OPEN classificados corretamente,
    0 falsos positivos em 85 exemplos de outros gestos.
    """
    ls = _kp(kps, KP.LEFT_SHOULDER,  conf_thr)
    le = _kp(kps, KP.LEFT_ELBOW,     conf_thr)
    lw = _kp(kps, KP.LEFT_WRIST,     conf_thr)
    rs = _kp(kps, KP.RIGHT_SHOULDER, conf_thr)
    re = _kp(kps, KP.RIGHT_ELBOW,    conf_thr)
    rw = _kp(kps, KP.RIGHT_WRIST,    conf_thr)

    if None in (ls, le, lw, rs, re, rw):
        return False

    sw = _shoulder_width(kps, bbox_w, conf_thr)

    # horiz para pessoa de frente: braço esq. vai para a DIREITA da imagem,
    # braço dir. vai para a ESQUERDA da imagem (ver _is_mudar_led)
    l_horiz = (lw[0] - ls[0]) / sw
    r_horiz = (rs[0] - rw[0]) / sw
    mx_horiz = max(l_horiz, r_horiz)
    mn_horiz = min(l_horiz, r_horiz)

    if mx_horiz < min_horiz:
        return False
    if mn_horiz < 0.1:   # lado menos estendido não pode apontar para o lado oposto
        return False

    l_vert = abs(lw[1] - ls[1]) / sw
    r_vert = abs(rw[1] - rs[1]) / sw
    if max(l_vert, r_vert) > max_vert:
        return False

    l_bend = 180.0 - _angle_between(ls[0], ls[1], le[0], le[1], lw[0], lw[1])
    r_bend = 180.0 - _angle_between(rs[0], rs[1], re[0], re[1], rw[0], rw[1])
    if max(l_bend, r_bend) > max_bend:
        return False

    return True

def _is_subir(kps: np.ndarray, bbox_w: float, conf_thr: float,
              min_ratio: float = UP_MIN_RATIO) -> bool:
    """
    Gesto SUBIR — pulsos acima da linha dos ombros, avaliado POR LADO
    INDIVIDUAL (pulso esquerdo vs ombro esquerdo, pulso direito vs ombro
    direito), não pela média dos dois lados.

    min_ratio exige uma margem mínima (em frações da largura dos ombros)
    além de simplesmente "acima" — isso evita falso positivo de ruído
    quando o pulso está só um pouco acima do ombro por acaso, com um braço
    relaxado. Calibrado para ficar acima do pior caso de repouso observado
    e abaixo do menor caso real de SUBIR observado (ver config.py).

    Exige os DOIS lados para disparar — só um braço levantado não conta
    como o gesto SUBIR da pessoa inteira.
    """
    ls = _kp(kps, KP.LEFT_SHOULDER,  conf_thr)
    rs = _kp(kps, KP.RIGHT_SHOULDER, conf_thr)
    lw = _kp(kps, KP.LEFT_WRIST,     conf_thr)
    rw = _kp(kps, KP.RIGHT_WRIST,    conf_thr)

    if None in (ls, rs, lw, rw):
        return False

    sw = _shoulder_width(kps, bbox_w, conf_thr)
    left_up  = (ls[1] - lw[1]) / sw >= min_ratio
    right_up = (rs[1] - rw[1]) / sw >= min_ratio
    return left_up and right_up


def _is_descer(kps: np.ndarray, bbox_w: float, conf_thr: float,
               min_ratio: float = DOWN_MIN_RATIO) -> bool:
    """
    Gesto DESCER — pulsos abaixo da linha dos quadris, avaliado POR LADO
    INDIVIDUAL, análogo a _is_subir.

    ATENÇÃO: min_ratio para DESCER foi calibrado SEM exemplo positivo real
    do gesto (não capturado no vídeo de calibração) — apenas para ficar
    fora da faixa observada de repouso. Validar com um vídeo real do
    gesto DOWN antes de confiar neste limiar em produção (ver config.py).
    """
    lh = _kp(kps, KP.LEFT_HIP,    conf_thr)
    rh = _kp(kps, KP.RIGHT_HIP,   conf_thr)
    lw = _kp(kps, KP.LEFT_WRIST,  conf_thr)
    rw = _kp(kps, KP.RIGHT_WRIST, conf_thr)

    if None in (lh, rh, lw, rw):
        return False

    sw = _shoulder_width(kps, bbox_w, conf_thr)
    left_down  = (lw[1] - lh[1]) / sw >= min_ratio
    right_down = (rw[1] - rh[1]) / sw >= min_ratio
    return left_down and right_down


# ── Classificador de gesto ──────────────────────────────────────────────────

class GestureAnalyzer:
    """Classifica o gesto de uma pessoa baseado nos keypoints."""

    def __init__(self, conf_thr: float = KP_CONF_THRESHOLD):
        self.conf_thr = conf_thr

    def classify(self, det: Detection) -> Gesture:
        kps    = det.kps
        bbox_w = float(det.bbox[2])
        thr    = self.conf_thr

        # FECHAR primeiro: depende do ÂNGULO/cruzamento dos antebraços, não
        # da largura de ombros como escala — funciona mesmo de perfil/ângulo
        # (confirmado em dados reais: frame 25 do testeB.mp4 tem o gesto
        # FECHAR genuíno com pose_reliable=False).
        if _is_fechar(kps, bbox_w, thr):
            return Gesture.FECHAR

        # Gate de confiabilidade para os DEMAIS gestos: todos eles usam
        # shoulder_width como escala de referência (vert_ratio, horiz_ratio,
        # wrist_shoulder_dy_norm, wrist_hip_dy_norm). Pessoa de perfil/costas
        # colapsa essa largura e qualquer ruído de pixel explode o ratio
        # (confirmado: frames 575-577 geraram falso DESCER com ratio até 6.6×
        # mesmo após a correção de limiar). Sem visão de frente, esses gestos
        # não têm como ser avaliados com confiança.
        if not _pose_reliable(kps, bbox_w, thr):
            return Gesture.REPOUSO

        # OPEN avaliado antes de MUDAR_LED: as condições de OPEN (braços
        # muito abertos lateralmente, max_horiz ≥ 1.0× largura dos ombros)
        # são um subconjunto estrito das de MUDAR_LED (que aceita qualquer
        # horiz ≥ 0). Testando LED primeiro (ordem original), casos reais de
        # OPEN eram classificados como LED. Calibrado com 31 exemplos reais:
        # OPEN antes de LED dá separação perfeita nos dados disponíveis.
        if _is_open(kps, bbox_w, thr):
            return Gesture.OPEN

        if _is_mudar_led(kps, bbox_w, thr):
            return Gesture.MUDAR_LED

        # Gestos de velocidade — avaliados por lado individual (ver _is_subir/_is_descer)
        if _is_subir(kps, bbox_w, thr):
            return Gesture.SUBIR

        if _is_descer(kps, bbox_w, thr):
            return Gesture.DESCER

        return Gesture.REPOUSO

    def ref_ys(self, det: Detection) -> tuple[Optional[float], Optional[float]]:
        """Retorna (shoulder_y, hip_y) para as linhas de referência visuais."""
        return (
            _mean_y(det.kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER], self.conf_thr),
            _mean_y(det.kps, [KP.LEFT_HIP,      KP.RIGHT_HIP],      self.conf_thr),
        )


# ── Coeficiente de velocidade ───────────────────────────────────────────────

def _reference_width(kps: np.ndarray, indices: list[int],
                     bbox_w: float, fallback_fraction: float,
                     conf_thr: float) -> Optional[float]:
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
    Coeficiente de velocidade [0, 1].
    Só faz sentido para SUBIR e DESCER; retorna 0 para os demais.
    """
    if gesture not in (Gesture.SUBIR, Gesture.DESCER):
        return 0.0

    kps = det.kps.copy()
    for joint_name in ("L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP",
                       "L_WRIST",    "R_WRIST"):
        x_f, y_f, _ = kp_filter.filtered_pos(joint_name, kps)
        idx = KALMAN_JOINTS[joint_name]
        kps[idx, 0] = x_f
        kps[idx, 1] = y_f

    bbox_w = float(det.bbox[2])

    if gesture == Gesture.SUBIR:
        ref_width = _reference_width(kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER],
                                     bbox_w, 0.4, conf_thr)
        ref_y     = _mean_y(kps, [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER], conf_thr)
        wrist_y   = _mean_y(kps, [KP.LEFT_WRIST,    KP.RIGHT_WRIST],    conf_thr)
        if None in (ref_width, ref_y, wrist_y):
            return 0.0
        dist = ref_y - wrist_y
    else:
        ref_width = _reference_width(kps, [KP.LEFT_HIP, KP.RIGHT_HIP],
                                     bbox_w, 0.3, conf_thr)
        ref_y     = _mean_y(kps, [KP.LEFT_HIP,   KP.RIGHT_HIP],   conf_thr)
        wrist_y   = _mean_y(kps, [KP.LEFT_WRIST, KP.RIGHT_WRIST], conf_thr)
        if None in (ref_width, ref_y, wrist_y):
            return 0.0
        dist = wrist_y - ref_y

    if dist <= 0.0:
        return 0.0
    return float(np.clip(dist / (SPEED_NORM_FACTOR * ref_width), 0.0, 1.0))


# ── Rastreamento de identidade entre frames ────────────────────────────────

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU entre dois bounding boxes no formato [x, y, w, h].
    Retorna valor em [0, 1].
    """
    ax1, ay1 = float(a[0]), float(a[1])
    ax2, ay2 = ax1 + float(a[2]), ay1 + float(a[3])
    bx1, by1 = float(b[0]), float(b[1])
    bx2, by2 = bx1 + float(b[2]), by1 + float(b[3])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = float(a[2]) * float(a[3])
    area_b = float(b[2]) * float(b[3])
    return inter / (area_a + area_b - inter)


def _match_detections(
    prev_bboxes: list[np.ndarray],
    curr_dets:   list[Detection],
    iou_thr:     float = 0.25,
) -> list[int]:
    """
    Associa cada detecção atual ao slot de identidade do frame anterior
    usando greedy matching por IoU decrescente.

    Retorna uma lista de tamanho len(curr_dets) onde cada elemento é o
    índice do slot (0..MAX_PERSONS-1) atribuído à detecção correspondente.
    Detecções sem match acima de iou_thr recebem o próximo slot livre.

    Slots são estáveis entre frames: a pessoa que estava no slot 0 no
    frame anterior continua no slot 0 mesmo que se mova na imagem, desde
    que a IoU das suas bboxes seja suficiente para o match.
    """
    n_slots = MAX_PERSONS
    n_dets  = len(curr_dets)

    # sem histórico (primeiro frame) → ordena por x como antes
    if not prev_bboxes:
        return list(range(min(n_dets, n_slots)))

    # matriz de IoU: slots (linhas) × detecções atuais (colunas)
    iou_mat = np.zeros((n_slots, n_dets), dtype=np.float32)
    for s, pb in enumerate(prev_bboxes):
        if pb is None:
            continue
        for d, det in enumerate(curr_dets):
            iou_mat[s, d] = _bbox_iou(pb, det.bbox)

    assigned_slot = [-1] * n_dets   # slot atribuído para cada det atual
    used_slots    = set()
    used_dets     = set()

    # greedy: pega o par (slot, det) com maior IoU primeiro
    flat = [(iou_mat[s, d], s, d)
            for s in range(n_slots)
            for d in range(n_dets)]
    flat.sort(reverse=True)

    for iou, s, d in flat:
        if iou < iou_thr:
            break
        if s in used_slots or d in used_dets:
            continue
        assigned_slot[d] = s
        used_slots.add(s)
        used_dets.add(d)

    # dets sem match → slots livres na ordem de chegada
    free_slots = [s for s in range(n_slots) if s not in used_slots]
    for d in range(n_dets):
        if assigned_slot[d] == -1 and free_slots:
            assigned_slot[d] = free_slots.pop(0)

    return assigned_slot


# ── Centroides ─────────────────────────────────────────────────────────────

def _det_centroid(det: Detection) -> tuple[float, float]:
    """Centro do bounding box: (cx, cy) em pixels da imagem original."""
    x, y, w, h = det.bbox
    return float(x + w / 2), float(y + h / 2)


# ── Agregação para múltiplas pessoas ───────────────────────────────────────

def classify_all(
    dets:        list[Detection],
    analyzers:   list[GestureAnalyzer],
    kp_filters:  list[KalmanPerson],
    dt:          float,
    conf_thr:    float = KP_CONF_THRESHOLD,
    prev_bboxes: list[np.ndarray] | None = None,
) -> tuple[list[Gesture], str, float, float, int, list[tuple[float, float]], list[np.ndarray]]:
    """
    Classifica gestos de todas as pessoas, mantendo identidade entre frames,
    e retorna os centroides de cada pessoa detectada.

    Parâmetros
    ----------
    prev_bboxes : lista de bboxes do frame anterior, indexada por slot.
        Passe `None` no primeiro frame; a partir do segundo, passe o
        `next_bboxes` retornado pela chamada anterior.

    Retorna
    -------
    gestures     : lista de Gesture por slot (tamanho n ≤ MAX_PERSONS)
    majority_state : gesto majoritário como string
    confidence   : fração de pessoas no gesto majoritário
    mean_speed   : velocidade média (só SUBIR/DESCER)
    count        : número de pessoas no gesto majoritário
    centroids    : lista de (cx, cy) por slot, em pixels da imagem original
    next_bboxes  : bboxes do frame atual por slot, para passar na próxima chamada
    """
    n = min(len(dets), MAX_PERSONS)

    # ── Rastreamento de identidade ──────────────────────────────────────────
    slot_indices = _match_detections(prev_bboxes or [], dets[:n])

    # reordenar dets de acordo com os slots atribuídos
    n_slots  = max(slot_indices) + 1 if slot_indices else 0
    slot_det = [None] * MAX_PERSONS  # det por slot
    for d_idx, s in enumerate(slot_indices):
        if s < MAX_PERSONS:
            slot_det[s] = dets[d_idx]

    gestures:   list[Gesture]              = []
    centroids:  list[tuple[float, float]]  = []
    next_bboxes: list[np.ndarray | None]   = [None] * MAX_PERSONS
    ordered_dets: list[Detection | None]   = []   # dets na ordem de slots, para visualização

    for s in range(MAX_PERSONS):
        det = slot_det[s]
        if det is None:
            continue
        kp_filters[s].update(det.kps, dt, conf_thr)
        gestures.append(analyzers[s].classify(det))
        centroids.append(_det_centroid(det))
        next_bboxes[s] = det.bbox.copy()
        ordered_dets.append(det)

    # ── Agregação de gesto ─────────────────────────────────────────────────
    majority_state = _majority(gestures)
    confidence     = _confidence(gestures, majority_state)

    target   = Gesture(majority_state)
    matching = [
        (slot_det[s], kp_filters[s], gestures[i])
        for i, s in enumerate(range(MAX_PERSONS))
        if i < len(gestures) and gestures[i] == target and slot_det[s] is not None
    ]
    count = len(matching)

    speed_gestures = (Gesture.SUBIR, Gesture.DESCER)
    if count > 0 and target in speed_gestures:
        mean_speed = float(np.mean([
            compute_speed_coeff(det, kf, g, conf_thr)
            for det, kf, g in matching
        ]))
    else:
        mean_speed = 0.0

    return gestures, majority_state, confidence, mean_speed, count, centroids, next_bboxes, ordered_dets


def _majority(gestures: list[Gesture]) -> str:
    if not gestures:
        return Gesture.REPOUSO.value
    counts = {g: gestures.count(g) for g in Gesture}
    return max(counts, key=counts.get).value


def _confidence(gestures: list[Gesture], majority: str) -> float:
    if not gestures:
        return 1.0
    target = Gesture(majority)
    return sum(1 for g in gestures if g == target) / len(gestures)
