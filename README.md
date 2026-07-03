# 🌂 Projeto Guarda-Chuva

# Resultado dos testes

[![Status do QA Central](https://github.com/Projeto-Guarda-Chuva/.github/actions/workflows/qa-t32.yml/badge.svg)](https://github.com/Projeto-Guarda-Chuva/.github/actions/workflows/qa-t32.yml)

# Detector de Gestos com YOLOv8-Pose

Este projeto implementa um sistema de detecção de gestos em tempo real utilizando um modelo YOLOv8-Pose no formato ONNX. Ele é capaz de identificar pessoas em um stream de vídeo, analisar a posição de seus pulsos em relação aos ombros e quadris para classificar gestos como "SUBIR", "DESCER" ou "REPOUSO", e calcular um coeficiente de velocidade para os movimentos.

## Como Funciona

O projeto tem dois modos de operação:

- **Modo Simples (`gesture_standalone.py`):** Ideal para testes rápidos. Um único programa abre sua webcam (ou um vídeo), detecta os gestos e mostra o resultado na tela. Tudo acontece em um só lugar.

- **Modo Avançado (dois programas):** Pensado para uso em produção.
  1.  `frame_interface.py`: Captura o vídeo de uma fonte de rede (stream MJPEG).
  2.  `gesture_detector.py`: Recebe esse vídeo e faz o trabalho pesado de detectar os gestos.

  Esse modo permite que a captura do vídeo e a análise rodem em computadores diferentes, o que é ótimo para otimizar o desempenho (por exemplo, usando uma GPU só para a detecção).

## O Que Cada Arquivo Faz

Os arquivos do projeto são organizados por função:

- **Executáveis Principais:**
  - `gesture_standalone.py`: Roda o projeto no **Modo Simples**.
  - `frame_interface.py`: Inicia a captura de vídeo no **Modo Avançado**.
  - `gesture_detector.py`: Inicia a detecção de gestos no **Modo Avançado**.

- **Lógica de Detecção:**
  - `detector.py`: Carrega o modelo de IA e encontra pessoas e seus pontos-chave (esqueleto).
  - `gesture_analyzer.py`: Analisa os pontos-chave para classificar o gesto ("SUBIR", "DESCER", etc.) e calcular a velocidade.
  - `kalman.py`: Usa um filtro para suavizar o movimento dos pontos-chave, tornando a detecção mais estável.

- **Utilitários e Configuração:**
  - `config.py`: Um lugar central para ajustar todas as configurações (sensibilidade, caminhos, etc.).
  - `state.py`: Salva o resultado final (gesto, velocidade) em um arquivo.
  - `visualizer.py`: Desenha os esqueletos, caixas e informações na tela.
  - `http_stream.py`: Conecta e recebe vídeo de uma fonte de rede.

- **Testes:**
  - `test_core.py`: Testes para garantir que a lógica principal está funcionando como esperado.

## Instalação

1.  **Crie um ambiente virtual:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    # venv\Scripts\activate    # No Windows
    ```

2.  **Instale as dependências:**
    (Crie um arquivo `requirements.txt` com o conteúdo abaixo e depois execute o `pip install`)

    **requirements.txt:**

    ```
    numpy
    opencv-python
    onnxruntime
    pytest
    ```

    **Comando de instalação:**

    ```bash
    pip install -r requirements.txt
    ```

    Para usar a aceleração por GPU, instale a versão apropriada do `onnxruntime-gpu`.

3.  **Obtenha um modelo:**
    Você precisará de um modelo de detecção de pose no formato `.onnx`, como o `yolov8n-pose.onnx`.

## Como Executar

### Modo Standalone (Recomendado para testes rápidos)

Execute o script `gesture_standalone.py`, passando o caminho para o modelo ONNX.

- **Para usar a webcam padrão (índice 0):**

  ```bash
  python gesture_standalone.py /caminho/para/seu/modelo.onnx
  ```

- **Para usar um arquivo de vídeo como fonte:**

  ```bash
  python gesture_standalone.py /caminho/para/seu/modelo.onnx --source /caminho/para/video.mp4
  ```

- **Para usar aceleração por GPU (CUDA e TensorRT):**
  ```bash
  python gesture_standalone.py /caminho/para/seu/modelo.onnx --gpu --tensorrt
  ```

### Modo de Dois Processos (Arquitetura Principal)

Este modo requer um stream de vídeo MJPEG rodando em algum lugar da sua rede (definido em `config.py`).

1.  **Inicie o Grupo 1 (Interface):**
    Em um terminal, execute `frame_interface.py`. Ele irá criar o _pipe_ e aguardar a conexão do Grupo 2.

    ```bash
    python frame_interface.py
    ```

2.  **Inicie o Grupo 2 (Detector):**
    Em outro terminal, execute `gesture_detector.py`, passando o caminho para o modelo.
    ```bash
    python gesture_detector.py /caminho/para/seu/modelo.onnx
    ```
    O detector irá se conectar ao _pipe_, começar a receber frames e exibir a janela com o resultado da análise.

## Como Rodar os Testes

Para garantir que a lógica principal (classificação de gestos, pré-processamento, etc.) está funcionando corretamente, você pode executar os testes automatizados usando `pytest`.

Na raiz do projeto, simplesmente execute o comando:

```bash
pytest
```
