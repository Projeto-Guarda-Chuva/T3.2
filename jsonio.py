"""
Utilitários de I/O para os arquivos JSON de estado/centroides.

Escrita atômica: grava em arquivo temporário no mesmo diretório e usa
os.replace() para substituir o arquivo final. Isso evita que um leitor
externo (ex: json_sender.py) leia o arquivo no meio de uma escrita —
os.replace() é atômico dentro do mesmo filesystem (POSIX/NTFS).
"""

import json
import logging
import os
import tempfile

log = logging.getLogger(__name__)


def atomic_write_json(path: str, data: dict) -> None:
    """Escreve `data` como JSON em `path` de forma atômica."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except OSError as e:
        log.warning("Falha ao escrever JSON em %s: %s", path, e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def read_json_safe(path: str) -> dict | None:
    """Lê e retorna o JSON em `path`. Retorna None se não existir ou for inválido."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
