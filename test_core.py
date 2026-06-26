"""
Testes unitários para a lógica principal do detector de gestos.
Foco: gesture_analyzer.py e kalman.py

Uso:
    pytest
"""

import sys
import os
import pytest
import numpy as np

# Adicionar o diretório do projeto ao path para encontrar os módulos
sys.path.insert(0, os.path.dirname(__file__))

from config import KP, KP_CONF_THRESHOLD, KALMAN_JOINTS
from detector import Detection
from gesture_analyzer import (
    Gesture, GestureAnalyzer, classify_all, compute_speed_coeff, _majority, _confidence
)
from kalman import KalmanJoint, KalmanPerson

# Mocks e Dados de Teste 

def create_mock_detection(kps_overrides: dict, x_offset: int = 0) -> Detection:
    """Cria um objeto Detection com keypoints mock."""
    # Keypoints em repouso, com confiança total
    kps = np.ones((KP.NUM, 3), dtype=np.float32)
    kps[:, 0] = 200 + x_offset  # x
    kps[:, 1] *= 250  # y

    # Posições padrão
    kps[KP.LEFT_SHOULDER,  1] = 200
    kps[KP.RIGHT_SHOULDER, 1] = 200
    kps[KP.LEFT_HIP,       1] = 300
    kps[KP.RIGHT_HIP,      1] = 300
    kps[KP.LEFT_WRIST,     1] = 250
    kps[KP.RIGHT_WRIST,    1] = 250

    # Largura dos ombros e quadris
    kps[KP.LEFT_SHOULDER,  0] = 150 + x_offset
    kps[KP.RIGHT_SHOULDER, 0] = 250 + x_offset
    kps[KP.LEFT_HIP,       0] = 160 + x_offset
    kps[KP.RIGHT_HIP,      0] = 240 + x_offset

    for kp_idx, (x, y, conf) in kps_overrides.items():
        kps[kp_idx] = [x, y, conf]

    bbox = [
        np.min(kps[:, 0]),
        np.min(kps[:, 1]),
        np.max(kps[:, 0]) - np.min(kps[:, 0]),
        np.max(kps[:, 1]) - np.min(kps[:, 1]),
    ]
    return Detection(bbox=np.array(bbox), score=0.9, kps=kps)


# Testes do GestureAnalyzer

@pytest.fixture
def analyzer():
    return GestureAnalyzer(conf_thr=KP_CONF_THRESHOLD)

def test_classify_repouso(analyzer):
    """Pulsos entre ombros e quadris."""
    det = create_mock_detection({})
    assert analyzer.classify(det) == Gesture.REPOUSO

def test_classify_subir(analyzer):
    """Pulsos acima dos ombros."""
    det = create_mock_detection({
        KP.LEFT_WRIST:  (150, 150, 1.0),
        KP.RIGHT_WRIST: (250, 150, 1.0),
    })
    assert analyzer.classify(det) == Gesture.SUBIR

def test_classify_descer(analyzer):
    """Pulsos abaixo dos quadris."""
    det = create_mock_detection({
        KP.LEFT_WRIST:  (150, 350, 1.0),
        KP.RIGHT_WRIST: (250, 350, 1.0),
    })
    assert analyzer.classify(det) == Gesture.DESCER

def test_classify_low_conf(analyzer):
    """Deve ignorar keypoints com baixa confiança."""
    det = create_mock_detection({
        KP.LEFT_WRIST:  (150, 150, 0.1), # Baixa confiança -> ignorado
        KP.RIGHT_WRIST: (250, 350, 1.0), # Abaixo do quadril
    })
    # Como um pulso está visível e abaixo do quadril, o gesto é DESCER
    assert analyzer.classify(det) == Gesture.DESCER

def test_classify_no_wrists(analyzer):
    """Sem pulsos visíveis, deve ser REPOUSO."""
    det = create_mock_detection({
        KP.LEFT_WRIST:  (150, 150, 0.1),
        KP.RIGHT_WRIST: (250, 150, 0.1),
    })
    assert analyzer.classify(det) == Gesture.REPOUSO


# Testes do Coeficiente de Velocidade

def test_speed_coeff_repouso():
    det = create_mock_detection({})
    kf = KalmanPerson()
    kf.update(det.kps, dt=0.03)
    assert compute_speed_coeff(det, kf, Gesture.REPOUSO) == 0.0

def test_speed_coeff_subir():
    # Pulsos bem acima dos ombros (y=100) vs ombros (y=200)
    # dist = 200 - 100 = 100
    # shoulder_width = 250 - 150 = 100
    # speed = 100 / (1.5 * 100) = 0.666...
    det = create_mock_detection({
        KP.LEFT_WRIST:  (150, 100, 1.0),
        KP.RIGHT_WRIST: (250, 100, 1.0),
    })
    kf = KalmanPerson()
    kf.update(det.kps, dt=0.03)
    speed = compute_speed_coeff(det, kf, Gesture.SUBIR)
    assert speed == pytest.approx(100.0 / (1.5 * 100.0))

def test_speed_coeff_descer():
    # Pulsos bem abaixo dos quadris (y=400) vs quadris (y=300)
    # dist = 400 - 300 = 100
    # hip_width = 240 - 160 = 80
    # speed = 100 / (1.5 * 80) = 0.833...
    det = create_mock_detection({
        KP.LEFT_WRIST:  (160, 400, 1.0),
        KP.RIGHT_WRIST: (240, 400, 1.0),
    })
    kf = KalmanPerson()
    kf.update(det.kps, dt=0.03)
    speed = compute_speed_coeff(det, kf, Gesture.DESCER)
    assert speed == pytest.approx(100.0 / (1.5 * 80.0))

def test_speed_coeff_clip():
    """Testa se a velocidade é limitada em 1.0."""
    # Distância muito grande
    det = create_mock_detection({
        KP.LEFT_WRIST:  (150, -100, 1.0),
        KP.RIGHT_WRIST: (250, -100, 1.0),
    })
    kf = KalmanPerson()
    kf.update(det.kps, dt=0.03)
    speed = compute_speed_coeff(det, kf, Gesture.SUBIR)
    assert speed == 1.0


# Testes da Agregação 

def test_majority_logic():
    assert _majority([Gesture.SUBIR, Gesture.SUBIR, Gesture.REPOUSO]) == "UP"
    assert _majority([Gesture.DESCER, Gesture.DESCER, Gesture.SUBIR]) == "DOWN"
    assert _majority([Gesture.REPOUSO, Gesture.REPOUSO]) == "REST"
    assert _majority([]) == "REST"
    # Em caso de empate, a implementação de max() pode escolher qualquer um.
    # O comportamento exato não é crítico, mas testamos um caso.
    assert _majority([Gesture.SUBIR, Gesture.DESCER]) in ("UP", "DOWN")

def test_confidence_logic():
    assert _confidence([Gesture.SUBIR, Gesture.SUBIR, Gesture.REPOUSO], "UP") == pytest.approx(2/3)
    assert _confidence([Gesture.DESCER, Gesture.DESCER, Gesture.SUBIR], "DOWN") == pytest.approx(2/3)
    assert _confidence([Gesture.DESCER, Gesture.DESCER, Gesture.SUBIR], "UP") == pytest.approx(1/3)
    assert _confidence([], "REST") == 1.0

def test_classify_all_aggregation():
    # Criar detecções com posições X diferentes para garantir uma ordenação estável.
    # A função `classify_all` ordena as detecções pela coordenada X do bbox.
    # Usamos x_offset para garantir que os bboxes não se sobreponham em x.
    # Para o gesto SUBIR, ambos os pulsos devem estar acima dos ombros para que a média funcione.
    det_up1 = create_mock_detection({
        KP.LEFT_WRIST: (160, 150, 1.0), KP.RIGHT_WRIST: (240, 150, 1.0)
    }, x_offset=0)
    det_up2 = create_mock_detection({
        KP.LEFT_WRIST: (360, 140, 1.0), KP.RIGHT_WRIST: (440, 140, 1.0)
    }, x_offset=200)
    det_rest = create_mock_detection({}, x_offset=450)

    dets = [det_up1, det_up2, det_rest]
    analyzers = [GestureAnalyzer() for _ in range(len(dets))]
    kp_filters = [KalmanPerson() for _ in range(len(dets))]

    gestures, state, conf, speed, count = classify_all(
        dets, analyzers, kp_filters, dt=0.03
    )

    assert gestures == [Gesture.SUBIR, Gesture.SUBIR, Gesture.REPOUSO]
    assert state == "UP"
    assert conf == pytest.approx(2/3)
    assert count == 2
    assert speed > 0.0


# Testes do Kalman 

def test_kalman_joint_init():
    kf = KalmanJoint()
    assert not kf.initialized
    kf.update(100, 200)
    assert kf.initialized
    assert np.allclose(kf.s, [100, 200, 0, 0, 0, 0])

def test_kalman_joint_predict_update():
    kf = KalmanJoint(q_acc=1.0, r_meas=1.0)
    kf.update(100, 200) # Posição inicial

    # 1. Prever
    kf.predict(dt=0.1)
    # Posição não deve mudar muito sem velocidade/aceleração inicial
    assert np.allclose(kf.pos, (100, 200))

    # 2. Atualizar com nova medição
    kf.update(110, 210)
    # O estado deve se mover em direção à medição
    pos_x, pos_y = kf.pos
    vel_x, vel_y = kf.vel
    assert 100 < pos_x < 110
    assert 200 < pos_y < 210
    # A velocidade deve se tornar positiva
    assert vel_x > 0
    assert vel_y > 0