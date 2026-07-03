import sys
import os
import pytest
import json
import numpy as np

# Adicionar o diretório do projeto ao path para encontrar os módulos
sys.path.insert(0, os.path.dirname(__file__))

from config import KP, KP_CONF_THRESHOLD, KALMAN_JOINTS
from detector import Detection
from gesture_analyzer import (
    Gesture, GestureAnalyzer, classify_all, compute_speed_coeff, _majority, _confidence
)
from kalman import KalmanJoint, KalmanPerson

# --- Carregamento dos Casos de Teste ---

def load_test_cases():
    path = os.path.join(os.path.dirname(__file__), "tests", "test_cases.json")
    with open(path, 'r') as f:
        return json.load(f)

TEST_CASES = load_test_cases()

def create_mock_detection(kps_overrides: dict, x_offset: int = 0) -> Detection:
    kps = np.ones((KP.NUM, 3), dtype=np.float32)
    kps[:, 0] = 200 + x_offset  # x
    kps[:, 1] *= 250  # y

    kps[KP.LEFT_SHOULDER,  1] = 200
    kps[KP.RIGHT_SHOULDER, 1] = 200
    kps[KP.LEFT_HIP,       1] = 300
    kps[KP.RIGHT_HIP,      1] = 300
    kps[KP.LEFT_WRIST,     1] = 250
    kps[KP.RIGHT_WRIST,    1] = 250

    kps[KP.LEFT_SHOULDER,  0] = 150 + x_offset
    kps[KP.RIGHT_SHOULDER, 0] = 250 + x_offset
    kps[KP.LEFT_HIP,       0] = 160 + x_offset
    kps[KP.RIGHT_HIP,      0] = 240 + x_offset

    for kp_idx_str, (x, y, conf) in kps_overrides.items():
        kp_idx = int(kp_idx_str)
        kps[kp_idx] = [x, y, conf]

    bbox = [
        np.min(kps[:, 0]),
        np.min(kps[:, 1]),
        np.max(kps[:, 0]) - np.min(kps[:, 0]),
        np.max(kps[:, 1]) - np.min(kps[:, 1]),
    ]
    return Detection(bbox=np.array(bbox), score=0.9, kps=kps)


# --- Testes Parametrizados ---

@pytest.fixture
def analyzer():
    return GestureAnalyzer(conf_thr=KP_CONF_THRESHOLD)

@pytest.mark.parametrize(
    "case",
    TEST_CASES["gesture_classification"],
    ids=[c["name"] for c in TEST_CASES["gesture_classification"]]
)
def test_gesture_classification(analyzer, case):
    """Testa a classificação de gestos a partir de casos definidos em JSON."""
    det = create_mock_detection(case["kps_overrides"])
    expected_gesture = Gesture(case["expected_gesture"])
    assert analyzer.classify(det) == expected_gesture

@pytest.mark.parametrize(
    "case",
    TEST_CASES["speed_coefficient"],
    ids=[c["name"] for c in TEST_CASES["speed_coefficient"]]
)
def test_speed_coefficient(case):
    """Testa o cálculo do coeficiente de velocidade a partir de casos definidos em JSON."""
    det = create_mock_detection(case["kps_overrides"])
    kf = KalmanPerson()
    kf.update(det.kps, dt=0.03)
    kf.update(det.kps, dt=0.03)
    gesture = Gesture(case["gesture"])
    
    speed = compute_speed_coeff(det, kf, gesture)
    assert speed == pytest.approx(case["expected_speed"], abs=1e-6)


# --- Testes de Lógica de Agregação e Kalman ---

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

@pytest.mark.parametrize(
    "case",
    TEST_CASES["aggregation"],
    ids=[c["name"] for c in TEST_CASES["aggregation"]]
)
def test_aggregation_scenarios(case):
    """Testa a lógica de agregação com múltiplas pessoas a partir do JSON."""
    dets = [
        create_mock_detection(p["kps_overrides"], p["x_offset"])
        for p in case["persons"]
    ]
    analyzers = [GestureAnalyzer() for _ in range(len(dets))]
    kp_filters = [KalmanPerson() for _ in range(len(dets))]

    gestures, state, conf, speed, count = classify_all(
        dets, analyzers, kp_filters, dt=0.03
    )
    
    expected_gestures = [Gesture(g) for g in case["expected_gestures"]]
    assert gestures == expected_gestures
    assert state == case["expected_state"]
    assert conf == pytest.approx(case["expected_confidence"], abs=1e-6)
    assert count == case["expected_count"]
    assert (speed > 0) == case["check_speed_positive"]


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