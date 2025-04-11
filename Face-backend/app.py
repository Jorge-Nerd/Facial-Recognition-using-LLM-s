import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from tqdm import tqdm  # Para barra de progresso
from utils import get_embedding

# Verificar disponibilidade de GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Carregue o modelo FaceNet pré-treinado
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

