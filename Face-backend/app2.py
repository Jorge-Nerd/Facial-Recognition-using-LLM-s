import streamlit as st
import cv2
import numpy as np
import os
import torch
import faiss
import sqlite3
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from datetime import datetime
from utils_models import GenderAgeModel
from utils import add_person, recognize_face, recognize_unknown_face, add_unknown_person
from db_manager import create_db_known, create_db_unknown

# Configurações iniciais
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
st.set_page_config(page_title="Sistema Facial Inteligente", layout="wide")
st.title("🎥 Sistema de Reconhecimento Facial com Cadastro em Tempo Real")

# Criar diretórios e BD
os.makedirs("known_faces", exist_ok=True)
create_db_known()
create_db_unknown()

# Carregamento de modelos
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(dev)
facemodel = YOLO('./yolov8n-face.pt')
gender_model = GenderAgeModel()
gender_model.load_state_dict(torch.load('./gender_age_model_new.pth', map_location=dev))
gender_model.eval()

# Conexões BD
conn_known = sqlite3.connect('known.db', check_same_thread=False)
conn_unknown = sqlite3.connect('unknown.db', check_same_thread=False)

# FAISS
DIM = 512
faiss_known_path = "faiss_known.index"
faiss_unknown_path = "faiss_unknown.index"

index_known = faiss.read_index(faiss_known_path) if os.path.exists(faiss_known_path) else faiss.IndexFlatL2(DIM)
index_unknown = faiss.read_index(faiss_unknown_path) if os.path.exists(faiss_unknown_path) else faiss.IndexFlatL2(DIM)

# Transformador de imagem
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Função para mostrar badge colorida ---
def format_person_display(name, age, gender, status):
    if status == "conhecido":
        color = "green"
        icon = "✅"
    elif status == "recorrente":
        color = "orange"
        icon = "♻️"
    else:
        color = "blue"
        icon = "🆕"
    return f"<span style='color:{color}; font-weight:bold;'>{icon} {name} | {age} anos | {gender}</span>"

st.sidebar.header("➕ Adicionar nova pessoa")
with st.sidebar.expander("Formulário de cadastro"):
    full_name = st.text_input("Nome completo")
    age = st.number_input("Idade", 0, 120)
    gender = st.selectbox("Gênero", ["Masculino", "Feminino"])

    photos = st.file_uploader("Seleciona 3 fotos da pessoa", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Adicionar Pessoa"):
        if not full_name.strip() or age == 0:
            st.warning("Preencha nome e idade.")
        elif len(photos) != 3:
            st.warning("Envie exatamente 3 fotos.")
        else:
            first_name = full_name.split()[0]
            person_dir = os.path.join("known_faces", first_name)
            os.makedirs(person_dir, exist_ok=True)
            for i, photo in enumerate(photos):
                path = os.path.join(person_dir, f"foto_{i+1}.jpg")
                with open(path, "wb") as f:
                    f.write(photo.read())
            result = add_person(full_name, age, gender, person_dir, index_known, conn_known)
            if result is None:
                st.error("Erro ao adicionar pessoa. Verifique as fotos.")
            else:
                faiss.write_index(index_known, faiss_known_path)
                st.success(f"Pessoa '{full_name}' adicionada com sucesso!")

st.markdown("---")
st.subheader("📷 Câmera em tempo real com reconhecimento")

if st.checkbox("Ativar câmera"):
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Erro ao acessar a câmera.")
                break

            frame = cv2.resize(frame, (960, 540))
            results = facemodel(frame, conf=0.5)[0]

            detected_faces_info = []

            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = frame[y1:y2, x1:x2]

                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_tensor = transform(face_pil).unsqueeze(0).to(dev)

                    with torch.no_grad():
                        pred_age, pred_gender = gender_model(face_tensor)
                    age_pred = int(pred_age.item())
                    gender_pred = "Feminino" if pred_gender.item() > 0.5 else "Masculino"

                    result = recognize_face(face, index_known, conn_known)
                    if result:
                        full_name, idade, genero, last_seen = result
                        status = "conhecido"
                        display_text = f"{full_name} | {idade} anos | {genero}"
                    else:
                        result_unknown = recognize_unknown_face(face, index_unknown, conn_unknown)
                        if result_unknown:
                            nome, idade, genero, last_seen = result_unknown
                            status = "recorrente"
                            display_text = f"{nome} (recorrente) | {idade} anos | {genero}"
                        else:
                            novo_nome = add_unknown_person(age_pred, gender_pred, face, index_unknown, conn_unknown)
                            status = "novo"
                            display_text = f"{novo_nome} (novo) | {age_pred} anos | {gender_pred}"

                    detected_faces_info.append({
                        "bbox": (x1, y1, x2, y2),
                        "text": display_text,
                        "status": status
                    })

                    # Desenhar retângulo sem texto no frame para manter limpo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Mostrar frame no Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Mostrar dados dos rostos detectados na coluna lateral
            with info_placeholder.container():
                st.markdown(f"### 🧠 Rostos detectados: {len(detected_faces_info)}")
                for face_info in detected_faces_info:
                    st.markdown(format_person_display(*face_info["text"].split("|"), face_info["status"]), unsafe_allow_html=True)

    except st.runtime.scriptrunner.StopException:
        pass

    finally:
        cap.release()
        faiss.write_index(index_known, faiss_known_path)
        faiss.write_index(index_unknown, faiss_unknown_path)
