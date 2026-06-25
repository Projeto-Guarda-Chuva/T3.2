"""
Backend de inferência ONNX Runtime.

Responsabilidades:
  - Carregar e executar o modelo YOLO11n-pose via ONNX Runtime
  - Pré-processar frames (letterbox + normalização)
  - Pós-processar saídas (DFL decode + NMS) → lista de Detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    CONF_THRESHOLD, NMS_IOU, KP, ORT_INTRA_THREADS, ORT_INTER_THREADS,
    TRT_CACHE_PATH, TRT_MAX_WORKSPACE,
)

log = logging.getLogger(__name__)

# ── Tipos de dados ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Uma pessoa detectada com bounding box e keypoints."""
    bbox:   np.ndarray          # [x, y, w, h] em coordenadas originais
    score:  float
    kps:    np.ndarray          # shape (17, 3) → [x, y, conf] por keypoint COCO


@dataclass
class PrepResult:
    """Resultado do pré-processamento de um frame."""
    tensor:   np.ndarray        # shape (1, 3, H, W), float32
    scale:    float             # fator de escala aplicado
    pad_left: int               # padding horizontal (px)
    pad_top:  int               # padding vertical (px)
    net_h:    int
    net_w:    int


# ── Pré-processamento ───────────────────────────────────────────────────────

def preprocess(frame: np.ndarray, net_h: int, net_w: int) -> PrepResult:
    """
    Letterbox + normalização [0,1] + transposição para CHW.
    Mantém proporção e preenche com cinza (114).
    """
    fh, fw = frame.shape[:2]
    scale  = min(net_h / fh, net_w / fw)
    nh, nw = int(round(fh * scale)), int(round(fw * scale))

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    padded = np.full((net_h, net_w, 3), 114, dtype=np.uint8)
    pad_top  = (net_h - nh) // 2
    pad_left = (net_w - nw) // 2
    padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

    rgb    = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1))[np.newaxis]  # (1,3,H,W)

    return PrepResult(tensor, scale, pad_left, pad_top, net_h, net_w)


# ── DFL decode ─────────────────────────────────────────────────────────────

def _dfl_decode(logits: np.ndarray) -> float:
    """Converte distribuição DFL (reg_max bins) em distância escalar."""
    logits  = logits - logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return float(np.dot(weights, np.arange(len(weights))))


# ── Pós-processamento ───────────────────────────────────────────────────────

def postprocess(
    outputs:     list[np.ndarray],
    prep:        PrepResult,
    orig_h:      int,
    orig_w:      int,
    conf_thr:    float = CONF_THRESHOLD,
    nms_iou:     float = NMS_IOU,
) -> list[Detection]:
    """
    Decodifica as saídas multi-stride do YOLO11n-pose.

    Formato de saída por stride: [1, 116, gridH, gridW]
      canais 0..63   → DFL box (4 × 16 bins)
      canal  64      → objectness logit
      canais 65..116 → keypoints (17 × 3: x, y, conf_logit)
    """
    REG_MAX = 16
    NUM_KP  = KP.NUM

    boxes, scores, kps_list = [], [], []

    for output in outputs:
        # output: (1, C, gH, gW) → (C, gH, gW)
        out = output[0]
        C, gH, gW = out.shape
        stride = prep.net_w / gW

        # Grids de ancora
        cx = (np.arange(gW) + 0.5) * stride   # (gW,)
        cy = (np.arange(gH) + 0.5) * stride   # (gH,)
        grid_cx, grid_cy = np.meshgrid(cx, cy)  # (gH, gW)

        # Objectness
        obj_conf = 1.0 / (1.0 + np.exp(-out[4 * REG_MAX]))  # (gH, gW)
        mask     = obj_conf >= conf_thr
        if not mask.any():
            continue

        gy_idx, gx_idx = np.where(mask)

        for gy, gx in zip(gy_idx, gx_idx):
            conf = float(obj_conf[gy, gx])
            ax, ay = float(grid_cx[gy, gx]), float(grid_cy[gy, gx])

            # DFL box decode
            dists = np.array([
                _dfl_decode(out[side * REG_MAX:(side + 1) * REG_MAX, gy, gx]) * stride
                for side in range(4)
            ])
            nx0, ny0 = ax - dists[0], ay - dists[1]
            nx1, ny1 = ax + dists[2], ay + dists[3]

            # Desletterbox → coords originais
            x0 = np.clip((nx0 - prep.pad_left) / prep.scale, 0, orig_w)
            y0 = np.clip((ny0 - prep.pad_top)  / prep.scale, 0, orig_h)
            x1 = np.clip((nx1 - prep.pad_left) / prep.scale, 0, orig_w)
            y1 = np.clip((ny1 - prep.pad_top)  / prep.scale, 0, orig_h)
            if x1 <= x0 or y1 <= y0:
                continue

            # Keypoints
            kps = np.zeros((NUM_KP, 3), dtype=np.float32)
            kp_base = 4 * REG_MAX + 1
            for k in range(NUM_KP):
                kx_r = float(out[kp_base + k * 3,     gy, gx])
                ky_r = float(out[kp_base + k * 3 + 1, gy, gx])
                kv_r = float(out[kp_base + k * 3 + 2, gy, gx])
                nkx  = kx_r * 2.0 * stride + ax - stride
                nky  = ky_r * 2.0 * stride + ay - stride
                kps[k, 0] = (nkx - prep.pad_left) / prep.scale
                kps[k, 1] = (nky - prep.pad_top)  / prep.scale
                kps[k, 2] = 1.0 / (1.0 + np.exp(-kv_r))

            boxes.append([x0, y0, x1 - x0, y1 - y0])
            scores.append(conf)
            kps_list.append(kps)

    if not boxes:
        return []

    # NMS via OpenCV
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, nms_iou)
    indices = indices.flatten() if len(indices) else []

    return [
        Detection(
            bbox  = np.array(boxes[i],    dtype=np.float32),
            score = scores[i],
            kps   = kps_list[i],
        )
        for i in indices
    ]


# ── ONNX Backend ────────────────────────────────────────────────────────────

class OnnxBackend:
    """
    Carrega e executa um modelo ONNX.
    Suporta CPU, CUDA EP e TensorRT EP.
    """

    def __init__(self, model_path: str, use_gpu: bool = False,
                 use_tensorrt: bool = False):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads       = ORT_INTRA_THREADS
        opts.inter_op_num_threads       = ORT_INTER_THREADS
        opts.graph_optimization_level   = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena       = True
        opts.enable_mem_pattern         = True

        providers = self._build_providers(use_gpu, use_tensorrt)
        log.info("Providers: %s", providers)

        self._session   = ort.InferenceSession(model_path, opts,
                                               providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        shape            = self._session.get_inputs()[0].shape
        self.net_h       = int(shape[2])
        self.net_w       = int(shape[3])
        log.info("Modelo carregado: %dx%d, %d saídas",
                 self.net_w, self.net_h, len(self._session.get_outputs()))

    @staticmethod
    def _build_providers(use_gpu: bool, use_tensorrt: bool) -> list:
        if not use_gpu:
            return ["CPUExecutionProvider"]

        providers = []
        if use_tensorrt:
            providers.append((
                "TensorrtExecutionProvider", {
                    "device_id":                    0,
                    "trt_max_workspace_size":       TRT_MAX_WORKSPACE,
                    "trt_fp16_enable":              True,
                    "trt_engine_cache_enable":      True,
                    "trt_engine_cache_path":        TRT_CACHE_PATH,
                },
            ))
        providers.append((
            "CUDAExecutionProvider", {
                "device_id":              0,
                "arena_extend_strategy":  "kNextPowerOfTwo",
                "gpu_mem_limit":          3 * 1024 ** 3,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            },
        ))
        providers.append("CPUExecutionProvider")
        return providers

    def run(self, prep: PrepResult) -> list[np.ndarray]:
        """Executa a inferência e retorna lista de arrays de saída."""
        return self._session.run(None, {self._input_name: prep.tensor})
