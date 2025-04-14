import cv2 
import cvzone
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from utils_models import GenderAgeModel
from utils import recognize_face, recognize_unknown_face, add_unknown_person
import sqlite3
import faiss
from db_manager import create_db_known, create_db_unknown

# Inicializar banco de dados
create_db_known()
create_db_unknown()

# Dispositivo (GPU se disponível)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Modelos
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
facemodel = YOLO('./yolov8n-face.pt')
model = GenderAgeModel()
model.load_state_dict(torch.load('./gender_age_model_new.pth', map_location=device))
model.eval()

# Banco de dados
conn_known = sqlite3.connect('known.db', check_same_thread=False)
conn_unknown = sqlite3.connect('unknown.db', check_same_thread=False)

# FAISS
dim = 512
# Diretórios dos índices
faiss_known_path = "faiss_known.index"
faiss_unknown_path = "faiss_unknown.index"

# Carrega FAISS dos conhecidos ou cria novo
if os.path.exists(faiss_known_path):
    index_known = faiss.read_index(faiss_known_path)
else:
    index_known = faiss.IndexFlatL2(dim)

# Carrega FAISS dos desconhecidos ou cria novo
if os.path.exists(faiss_unknown_path):
    index_unknown = faiss.read_index(faiss_unknown_path)
else:
    index_unknown = faiss.IndexFlatL2(dim)


# Transformação da face
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Captura de vídeo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1100, 720))
    results = facemodel(frame, conf=0.5)[0]

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            face = frame[y1:y2, x1:x2]

            # Preprocessar a face
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Predizer idade e gênero
            with torch.no_grad():
                age, gender = model(face_tensor)
            age_pred = int(age.item())
            gender_pred = 'Feminino' if gender.item() > 0.5 else 'Masculino'

            # Reconhecimento facial
            result = recognize_face(face, index_known, conn_known)
            if result:
                full_name, idade, genero = result
                texto = f"{full_name} | {idade} anos | {genero}"
            else:
                result_unknown = recognize_unknown_face(face, index_unknown, conn_unknown)
                if result_unknown:
                    nome, idade, genero = result_unknown
                    texto = f"{nome} (recorrente) | {idade} anos | {genero}"
                else:
                    novo_nome = add_unknown_person(age_pred, gender_pred, face, index_unknown, conn_unknown)
                    texto = f"{novo_nome} (novo) | {age_pred} anos | {gender_pred}"

            # Mostrar caixa e texto
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
            cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
faiss.write_index(index_known, faiss_known_path)
faiss.write_index(index_unknown, faiss_unknown_path)
cv2.destroyAllWindows()
