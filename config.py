# config.py (modificado)
"""
Configurações centralizadas do sistema de detecção de gestos.
Altere aqui para ajustar comportamento sem tocar na lógica.
"""

# ── Rede / pipeline ────────────────────────────────────────────────────────
PIPE_PATH            = "/tmp/gesture_frame_pipe"
STATE_FILE           = "/tmp/gesture_state.json"
HTTP_URL             = "http://100.88.0.12:5000"  # Mantido por compatibilidade
FPS_LIMIT            = 30
FRAME_QUEUE_MAX_SIZE = 3
READ_TIMEOUT_MS      = 1000        # timeout de leitura do pipe (ms)
STATE_UPDATE_INTERVAL = 0.050      # intervalo de escrita do JSON (segundos)

# ── Câmera local (Jetson Orin Nano) ──────────────────────────────────────
CAMERA_DEVICE        = 0           # 0 = /dev/video0, 1 = /dev/video1, etc.
CAMERA_WIDTH         = 1280        # Largura de captura
CAMERA_HEIGHT        = 720         # Altura de captura
CAMERA_FPS           = 30          # FPS de captura
CAMERA_BUFFERSIZE    = 1           # Buffer mínimo para baixa latência

# ── Detecção / NMS ─────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.40              # confiança mínima de detecção
KP_CONF_THRESHOLD = 0.40          # confiança mínima de keypoint
NMS_IOU = 0.45                     # IoU para NMS
MAX_PERSONS = 4                    # máximo de pessoas rastreadas por frame

# ── Kalman (DWPA 6-state) ──────────────────────────────────────────────────
KALMAN_Q_ACC  = 30.0               # ruído de processo na aceleração (px/s²)
                                   # alto = acompanha movimentos bruscos melhor
KALMAN_R_MEAS = 15.0                # variância da medição (px²)
                                   # baixo = confia mais no YOLO
KALMAN_P_INIT = 50.0               # covariância inicial da diagonal
KALMAN_DT_MAX = 0.5                # dt máximo entre frames (s) — evita explosão

# ── Coeficiente de velocidade ──────────────────────────────────────────────
SPEED_NORM_FACTOR = 1.5            # distância (em larguras de ombro/quadril)
                                   # que corresponde a speed = 1.0

# ── ONNX Runtime ──────────────────────────────────────────────────────────
ORT_INTRA_THREADS  = 6             # threads intra-op (Orin Nano tem 6 cores)
ORT_INTER_THREADS  = 2             # threads inter-op
TRT_CACHE_PATH     = "/tmp/trt_cache"
TRT_MAX_WORKSPACE  = 1 << 30       # 1 GB

# ── Visualização ───────────────────────────────────────────────────────────
WINDOW_NAME   = "Gesture Detector"
WINDOW_W      = 1280
WINDOW_H      = 720
SPEED_BAR_W   = 200
SPEED_BAR_H   = 14
LOG_INTERVAL  = 30                 # logar a cada N frames

# ── Índices COCO dos keypoints ─────────────────────────────────────────────
class KP:
    NOSE           = 0
    LEFT_EYE       = 1
    RIGHT_EYE      = 2
    LEFT_EAR       = 3
    RIGHT_EAR      = 4
    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW     = 7
    RIGHT_ELBOW    = 8
    LEFT_WRIST     = 9
    RIGHT_WRIST    = 10
    LEFT_HIP       = 11
    RIGHT_HIP      = 12
    LEFT_KNEE      = 13
    RIGHT_KNEE     = 14
    LEFT_ANKLE     = 15
    RIGHT_ANKLE    = 16
    NUM            = 17

SKELETON = [
    (KP.NOSE, KP.LEFT_EYE),    (KP.NOSE, KP.RIGHT_EYE),
    (KP.LEFT_EYE, KP.LEFT_EAR),(KP.RIGHT_EYE, KP.RIGHT_EAR),
    (KP.LEFT_SHOULDER,  KP.RIGHT_SHOULDER),
    (KP.LEFT_SHOULDER,  KP.LEFT_ELBOW),
    (KP.LEFT_ELBOW,     KP.LEFT_WRIST),
    (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW),
    (KP.RIGHT_ELBOW,    KP.RIGHT_WRIST),
    (KP.LEFT_SHOULDER,  KP.LEFT_HIP),
    (KP.RIGHT_SHOULDER, KP.RIGHT_HIP),
    (KP.LEFT_HIP,       KP.RIGHT_HIP),
    (KP.LEFT_HIP,       KP.LEFT_KNEE),
    (KP.LEFT_KNEE,      KP.LEFT_ANKLE),
    (KP.RIGHT_HIP,      KP.RIGHT_KNEE),
    (KP.RIGHT_KNEE,     KP.RIGHT_ANKLE),
]

# Joints monitorados pelo Kalman e seus índices COCO correspondentes
KALMAN_JOINTS = {
    "L_SHOULDER": KP.LEFT_SHOULDER,
    "R_SHOULDER": KP.RIGHT_SHOULDER,
    "L_ELBOW":    KP.LEFT_ELBOW,
    "R_ELBOW":    KP.RIGHT_ELBOW,
    "L_HIP":      KP.LEFT_HIP,
    "R_HIP":      KP.RIGHT_HIP,
    "L_WRIST":    KP.LEFT_WRIST,
    "R_WRIST":    KP.RIGHT_WRIST,
}