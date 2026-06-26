# test_camera.py
"""
Script de teste para verificar a câmera local.
Útil para debug antes de rodar o sistema completo.
"""

import cv2
import time
import sys

from config import CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

def test_camera(device: int = CAMERA_DEVICE):
    print(f"Testando câmera /dev/video{device}...")
    
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"❌ Erro: não foi possível abrir /dev/video{device}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    print(f"✅ Câmera aberta: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    
    # Capturar alguns frames para teste
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  Frame {i+1}: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"  ❌ Frame {i+1} falhou")
    
    # Mostrar um frame se disponível
    if ret and frame is not None:
        cv2.imshow("Teste Câmera", frame)
        print("Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cap.release()
    return True

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)