"""
Configurações centralizadas do sistema de detecção de gestos.
Altere aqui para ajustar comportamento sem tocar na lógica.
"""

# ── Rede / pipeline ────────────────────────────────────────────────────────
PIPE_PATH            = "/tmp/gesture_frame_pipe"
STATE_FILE           = "/tmp/gesture_state.json"      # JSON de estado/gesto (modificável)
CENTROIDS_FILE       = "/tmp/gesture_centroids.json"  # JSON de centroides das pessoas (modificável)
HTTP_URL             = "http://172.18.4.56:443"
FPS_LIMIT            = 30
FRAME_QUEUE_MAX_SIZE = 3
READ_TIMEOUT_MS      = 1000        # timeout de leitura do pipe (ms)
STATE_UPDATE_INTERVAL = 0.050      # intervalo de escrita do JSON (segundos)

# ── Envio HTTP dos JSONs (state + centroides) ──────────────────────────────
JSON_SEND_URL      = "http://172.18.4.56:8180/gesture_update"  # endpoint de destino
JSON_SEND_INTERVAL = 0.2           # segundos entre envios
JSON_SEND_TIMEOUT  = 2.0           # timeout de conexão/leitura (s)

# ── Detecção / NMS ─────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.40              # confiança mínima de detecção
KP_CONF_THRESHOLD = 0.40          # confiança mínima de keypoint
NMS_IOU = 0.45                     # IoU para NMS
MAX_PERSONS = 4                    # máximo de pessoas rastreadas por frame

# ── Kalman (DWPA 6-state) ──────────────────────────────────────────────────
KALMAN_Q_ACC  = 80.0               # ruído de processo na aceleração (px/s²)
                                   # alto = acompanha movimentos bruscos melhor
KALMAN_R_MEAS = 2.0                # variância da medição (px²)
                                   # baixo = confia mais no YOLO
KALMAN_P_INIT = 50.0               # covariância inicial da diagonal
KALMAN_DT_MAX = 0.5                # dt máximo entre frames (s) — evita explosão

# ── Coeficiente de velocidade ──────────────────────────────────────────────
SPEED_NORM_FACTOR = 1.5            # distância (em larguras de ombro/quadril)
                                   # que corresponde a speed = 1.0

# ── Gestos SUBIR / DESCER ──────────────────────────────────────────────────
# Avaliados POR LADO INDIVIDUAL: pulso esquerdo vs ombro/quadril esquerdo,
# pulso direito vs ombro/quadril direito — não pela média dos dois lados.
# Cada lado precisa cruzar sua própria margem para contar como SUBIR/DESCER
# naquele lado; o gesto da PESSOA é considerado SUBIR/DESCER apenas se os
# DOIS lados cruzarem (evita falso positivo de um braço relaxado assimétrico).
#
# Calibrado em 26/06/2026 com gesture_labels.json (vídeo testeB.mp4, 116
# rótulos manuais cruzados com as métricas reais do pipeline):
#   - UP real (gesto intencional, por lado): wrist_shoulder_dy_norm ∈ [0.49, 2.20]
#   - REST (braço relaxado, por lado):       wrist_shoulder_dy_norm ∈ [-2.31, 0.35]
#   → UP_MIN_RATIO = 0.40 fica na margem livre entre REST (máx 0.35) e UP (mín 0.49)
#
#   - DOWN real: NENHUM exemplo capturado neste vídeo (ator não chegou a
#     fazer o gesto). NÃO HÁ DADO POSITIVO PARA CALIBRAR.
#   - REST (braço relaxado, por lado):       wrist_hip_dy_norm ∈ [-1.80, 0.79]
#     Isto é: braço relaxado já coloca o pulso até 0.79× a largura do ombro
#     ABAIXO do quadril (ex: mãos cruzadas na frente da barriga). O limiar
#     antigo (wrist_y > hip_y, equivalente a > 0.0) ESTAVA DENTRO dessa faixa
#     de repouso — por isso disparava DOWN para braços apenas relaxados.
#   → DOWN_MIN_RATIO = 1.00 escolhido para ficar comprovadamente acima do
#     pior caso de repouso observado (0.79), com folga de ~25%. Ainda assim,
#     ESTE VALOR NÃO FOI VALIDADO CONTRA UM DOWN REAL — recomendo gravar um
#     vídeo curto com o gesto DOWN intencional e reprocessar para confirmar
#     ou ajustar este número antes de confiar nele em produção.
UP_MIN_RATIO   = 0.40   # fração da largura dos ombros — pulso acima do ombro
DOWN_MIN_RATIO = 0.40   # fração da largura dos ombros — pulso abaixo do quadril



# ── Gesto OPEN (T-pose) ────────────────────────────────────────────────────
# RECALIBRADO em 26/06/2026 com gesture_labels.json (31 exemplos reais de
# OPEN rotulados manualmente em testeB.mp4).
#
# OPEN_MIN_HORIZ_RATIO: agora exigido no MÁXIMO dos dois lados, não em
#   ambos individualmente. Em poses assimétricas (oclusão parcial de um
#   braço), um lado pode estar bem abaixo de 1.0 mas o outro bem acima —
#   a lógica nova aceita isso desde que o lado fechado ainda seja >= 0.1
#   (não aponte para o lado errado) e o lado aberto seja >= OPEN_MIN_HORIZ_RATIO.
#   Calibrado: 31/31 OPEN detectados, 0 falsos positivos em 85 outros.
#
# OPEN_MAX_VERT_RATIO: avaliado pelo MÁXIMO dos dois lados. Aumentado de
#   0.5 para 0.9 para cobrir poses com um braço ligeiramente fora do nível
#   horizontal (especialmente sob oclusão parcial).
#
# OPEN_MAX_ELBOW_BEND: avaliado pelo MÁXIMO dos dois lados. Aumentado de
#   30° para 70° para cobrir braços parcialmente dobrados por oclusão.
#
# Além disso, o sinal horizontal foi corrigido (bug histórico): pessoa de
#   frente para a câmera tem o braço esquerdo dela projetado para a DIREITA
#   da imagem — ver _is_open() em gesture_analyzer.py para detalhes.
OPEN_MIN_HORIZ_RATIO = 0.9    # max(L,R) — braço mais estendido deve superar este valor
OPEN_MAX_VERT_RATIO  = 0.8    # max(L,R) — tolerância vertical aumentada
OPEN_MAX_ELBOW_BEND  = 70.0   # max(L,R) — tolerância de dobramento aumentada
# ── Gesto FECHAR (X com os antebraços) ────────────────────────────────────
# RECALIBRADO em 26/06/2026 com gesture_labels.json (24 exemplos reais de
# CLOSE rotulados manualmente em testeB.mp4). A versão anterior exigia
# "cruzamento" estrito de sinal dos pulsos (um pulso passando por cima do
# outro) — confirmado que essa condição falha em 100% dos casos reais
# (24/24), porque na prática os pulsos ficam próximos sem de fato cruzar.
#
# Nova lógica: cada antebraço (cotovelo→pulso) deve apontar para cima/
# diagonal-cima (ângulo positivo, ver kalman.py/_vec_angle_deg: 90° = reto
# para cima), E os pulsos devem estar próximos um do outro. A combinação
# discrimina de UP (que também levanta os antebraços, mas com os PULSOS
# AFASTADOS, um de cada lado).
#
# CLOSE_ANG_MIN / CLOSE_ANG_MAX: faixa aceitável do ângulo de cada
#   antebraço (graus, 90°=reto p/ cima). Fora dessa faixa o antebraço está
#   apontando para baixo (braço relaxado) ou quase horizontal (LED/OPEN).
#   Calibrado: exemplos reais de CLOSE ficaram em L∈[86°,143°] R∈[28°,117°];
#   usamos uma faixa comum [20°,150°] com folga para ambos os lados.
#
# CLOSE_MAX_WRIST_DIST: distância máxima entre os pulsos normalizada pela
#   largura dos ombros. Calibrado: CLOSE real ficou em [0.04, 1.14]; o menor
#   caso real de UP (gesto mais próximo) ficou em 1.19. A margem entre os
#   dois é estreita (~0.05) — 1.15 fica no meio do gap observado, mas vale
#   revalidar com mais dados se houver falsos positivos/negativos em campo.
CLOSE_ANG_MIN         = 20.0   # graus — abaixo disso, antebraço quase horizontal
CLOSE_ANG_MAX         = 150.0  # graus — acima disso, antebraço quase horizontal (outro lado)
CLOSE_MAX_WRIST_DIST  = 1.15   # fração da largura dos ombros

# ── Gesto MUDAR_LED (braços paralelos estendidos à frente) ────────────────
# Câmera posicionada acima e à frente: braços estendidos aparecem como
# segmentos quase horizontais, pequenos (profundidade), paralelos.
#
# LED_MAX_VERT_RATIO: desvio vertical máximo do pulso em relação ao ombro,
#   normalizado pela largura dos ombros. Controla o quanto o braço pode
#   estar inclinado para cima/baixo.
#   Ex: 0.5 → pulso pode estar até 50% da largura dos ombros acima/abaixo.
#
# LED_MIN_HORIZ_RATIO: extensão horizontal mínima do pulso além do ombro,
#   normalizada pela largura dos ombros.
#   Ex: 0.1 → pulso precisa estar ao menos 10% da largura para fora do ombro.
#   Use valor pequeno ou 0 se a câmera estiver muito de cima (braços somem).
#
# LED_MAX_ELBOW_BEND: curvatura máxima do cotovelo (graus). 0 = totalmente
#   estendido; 30 = permite braço um pouco dobrado.
LED_MAX_VERT_RATIO   = 0.5    # fração da largura dos ombros
LED_MIN_HORIZ_RATIO  = 0.0    # fração da largura dos ombros
LED_MAX_ELBOW_BEND   = 40.0   # graus — curvatura máxima do cotovelo
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
