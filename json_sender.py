"""
Enviador de JSONs — Grupo 3 (opcional)
=======================================
Lê periodicamente os arquivos state.json e centroids.json (escritos por
gesture_standalone.py ou gesture_detector.py) e envia o conteúdo via HTTP
POST para o endereço configurado em config.JSON_SEND_URL.

Roda como processo independente, em paralelo aos scripts principais —
não importa nada deles, apenas lê os arquivos em disco. Pode ser
iniciado e encerrado a qualquer momento sem afetar a detecção.

Payload enviado (JSON):
    {
        "timestamp": <float, epoch em segundos>,
        "state":     {... conteúdo de STATE_FILE ...} | null,
        "centroids": {... conteúdo de CENTROIDS_FILE ...} | null
    }

Uso:
    python json_sender.py
    python json_sender.py --url http://192.168.1.50:5001/gesture_update
    python json_sender.py --interval 0.5 --timeout 3
"""

import argparse
import json
import logging
import signal
import sys
import time
import urllib.error
import urllib.request

from config import (
    STATE_FILE, CENTROIDS_FILE,
    JSON_SEND_URL, JSON_SEND_INTERVAL, JSON_SEND_TIMEOUT,
)
from jsonio import read_json_safe

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("json_sender")

LOG_INTERVAL_SENDS = 50   # logar estatísticas a cada N envios (sucesso+falha)


# ── Envio HTTP ────────────────────────────────────────────────────────────

def send_payload(url: str, payload: dict, timeout: float) -> bool:
    """Envia `payload` como JSON via POST. Retorna True em caso de sucesso (2xx)."""
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        log.warning("Servidor retornou erro HTTP %d: %s", e.code, e.reason)
        return False
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        log.warning("Falha ao enviar para %s: %s", url, e)
        return False


# ── Loop principal ───────────────────────────────────────────────────────

def run(url: str = JSON_SEND_URL, interval: float = JSON_SEND_INTERVAL,
        timeout: float = JSON_SEND_TIMEOUT) -> int:
    running = True

    def _shutdown(sig, _frame):
        nonlocal running
        log.info("Encerrando (sinal %d)...", sig)
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Destino: %s  intervalo=%.2fs  timeout=%.1fs", url, interval, timeout)
    log.info("Lendo: %s  e  %s", STATE_FILE, CENTROIDS_FILE)

    sent, failed  = 0, 0
    last_frame_id = None
    t_start       = time.monotonic()

    while running:
        t_loop = time.monotonic()

        state     = read_json_safe(STATE_FILE)
        centroids = read_json_safe(CENTROIDS_FILE)

        if state is None and centroids is None:
            # Arquivos ainda não existem — os scripts principais podem
            # não ter iniciado ainda. Não é um erro fatal, só espera.
            time.sleep(interval)
            continue

        # Evita reenviar o mesmo frame repetidamente quando o pipeline de
        # visão está mais lento que o intervalo de envio configurado.
        curr_frame_id = centroids.get("frame") if centroids else None
        if curr_frame_id is not None and curr_frame_id == last_frame_id:
            elapsed = time.monotonic() - t_loop
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            continue

        payload = {
            "timestamp": time.time(),
            "state":     state,
            "centroids": centroids,
        }

        if send_payload(url, payload, timeout):
            sent += 1
            last_frame_id = curr_frame_id
        else:
            failed += 1

        total = sent + failed
        if total % LOG_INTERVAL_SENDS == 0:
            log.info("Enviados: %d  Falhas: %d", sent, failed)

        elapsed   = time.monotonic() - t_loop
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    elapsed = time.monotonic() - t_start
    log.info("Encerrado — %d enviados, %d falhas em %.1f s", sent, failed, elapsed)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enviador HTTP dos JSONs de estado/centroides")
    p.add_argument("--url", default=JSON_SEND_URL,
                   help=f"Endpoint HTTP de destino (padrão: {JSON_SEND_URL})")
    p.add_argument("--interval", type=float, default=JSON_SEND_INTERVAL,
                   help=f"Intervalo entre envios em segundos (padrão: {JSON_SEND_INTERVAL})")
    p.add_argument("--timeout", type=float, default=JSON_SEND_TIMEOUT,
                   help=f"Timeout de conexão/leitura em segundos (padrão: {JSON_SEND_TIMEOUT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(args.url, args.interval, args.timeout))
