import struct

import cv2
import numpy as np
import pytest

from detector import Detection, _dfl_decode, preprocess
from frame_interface import encode_frame
from gesture_analyzer import (Gesture, GestureAnalyzer, KalmanPerson,
                              _confidence, _majority, classify_all,
                              compute_speed_coeff)


def make_kps(shoulder_y=50, hip_y=150, wrist_y=30, conf=0.9):
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[5] = [40.0, float(shoulder_y), conf]  # LEFT_SHOULDER
    kps[6] = [80.0, float(shoulder_y), conf]  # RIGHT_SHOULDER
    kps[11] = [40.0, float(hip_y), conf]      # LEFT_HIP
    kps[12] = [80.0, float(hip_y), conf]      # RIGHT_HIP
    kps[9] = [30.0, float(wrist_y), conf]      # LEFT_WRIST
    kps[10] = [90.0, float(wrist_y), conf]     # RIGHT_WRIST
    return kps


def test_gesture_analyzer_classify_rest_when_wrist_missing():
    det = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                    np.zeros((17, 3), dtype=np.float32))
    analyzer = GestureAnalyzer()
    assert analyzer.classify(det) == Gesture.REPOUSO


@pytest.mark.parametrize(
    ("shoulder_y", "hip_y", "wrist_y", "expected"),
    [
        (100, 200, 50, Gesture.SUBIR),
        (50, 100, 150, Gesture.DESCER),
        (50, 150, 100, Gesture.REPOUSO),
    ],
)
def test_gesture_analyzer_classify_variants(shoulder_y, hip_y, wrist_y, expected):
    analyzer = GestureAnalyzer()
    det = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                    make_kps(shoulder_y=shoulder_y, hip_y=hip_y, wrist_y=wrist_y))
    assert analyzer.classify(det) == expected


def test_gesture_analyzer_ref_ys_returns_shoulder_and_hip_mean():
    analyzer = GestureAnalyzer()
    det = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                    make_kps(shoulder_y=80, hip_y=140, wrist_y=30))

    shoulder_y, hip_y = analyzer.ref_ys(det)
    assert shoulder_y == pytest.approx(80.0)
    assert hip_y == pytest.approx(140.0)


def test_compute_speed_coeff_returns_zero_for_rest():
    det = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                    make_kps(shoulder_y=50, hip_y=150, wrist_y=30))
    kp_filter = KalmanPerson()
    assert compute_speed_coeff(det, kp_filter, Gesture.REPOUSO) == 0.0


def test_compute_speed_coeff_norms_for_subir_and_descend():
    det_up = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                       make_kps(shoulder_y=100, hip_y=200, wrist_y=30))
    det_down = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                         make_kps(shoulder_y=50, hip_y=120, wrist_y=160))
    kp_filter = KalmanPerson()

    speed_up = compute_speed_coeff(det_up, kp_filter, Gesture.SUBIR)
    speed_down = compute_speed_coeff(det_down, kp_filter, Gesture.DESCER)

    assert 0.0 < speed_up <= 1.0
    assert 0.0 < speed_down <= 1.0


def test_compute_speed_coeff_returns_zero_when_reference_unavailable():
    det = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                    make_kps(shoulder_y=100, hip_y=200, wrist_y=110, conf=0.0))
    kp_filter = KalmanPerson()

    assert compute_speed_coeff(det, kp_filter, Gesture.SUBIR) == 0.0


def test_majority_and_confidence_helpers():
    gestures = [Gesture.SUBIR, Gesture.SUBIR, Gesture.DESCER]
    assert _majority(gestures) == "UP"
    assert _confidence(gestures, "UP") == pytest.approx(2.0 / 3.0)
    assert _majority([]) == "REST"
    assert _confidence([], "REST") == 1.0


def test_classify_all_aggregates_majority_confidence_and_count():
    det1 = Detection(np.array([0, 0, 100, 100], dtype=np.float32), 1.0,
                     make_kps(shoulder_y=100, hip_y=200, wrist_y=30))
    det2 = Detection(np.array([120, 0, 100, 100], dtype=np.float32), 1.0,
                     make_kps(shoulder_y=100, hip_y=200, wrist_y=30))
    analyzers = [GestureAnalyzer(), GestureAnalyzer()]
    kp_filters = [KalmanPerson(), KalmanPerson()]

    gestures, majority_state, confidence, mean_speed, count = classify_all(
        [det1, det2], analyzers, kp_filters, dt=0.0,
    )

    assert gestures == [Gesture.SUBIR, Gesture.SUBIR]
    assert majority_state == "UP"
    assert confidence == 1.0
    assert count == 2
    assert mean_speed > 0.0


def test_classify_all_returns_rest_when_no_detections():
    gestures, majority_state, confidence, mean_speed, count = classify_all(
        [], [], [], dt=0.0,
    )

    assert gestures == []
    assert majority_state == "REST"
    assert confidence == 1.0
    assert mean_speed == 0.0
    assert count == 0


def test_detector_preprocess_letterboxes_and_normalizes():
    frame = np.full((40, 80, 3), 128, dtype=np.uint8)
    prep = preprocess(frame, net_h=64, net_w=64)

    assert prep.tensor.shape == (1, 3, 64, 64)
    assert prep.net_h == 64
    assert prep.net_w == 64
    assert prep.scale == 0.8
    assert prep.pad_left == 0
    assert prep.pad_top == 16
    assert np.all(prep.tensor >= 0.0) and np.all(prep.tensor <= 1.0)


def test_dfl_decode_computes_weighted_expectation():
    logits = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    decoded = _dfl_decode(logits)
    assert 2.0 < decoded < 3.0


def test_frame_interface_encode_frame_size_and_payload():
    frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    payload = encode_frame(frame, frame_id=13)
    expected_len = struct.calcsize("<IIIII Q") + frame.nbytes

    assert len(payload) == expected_len
    assert payload[-frame.nbytes:] == frame.tobytes()


def test_frame_interface_encode_frame_header():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = encode_frame(frame, frame_id=7)

    header = payload[:struct.calcsize("<IIIII Q")]
    magic, frame_id, width, height, channels, timestamp = struct.unpack(
        "<IIIII Q", header)

    assert magic == 0x47525544
    assert frame_id == 7
    assert width == 2
    assert height == 2
    assert channels == 3
    assert payload.endswith(frame.tobytes())
