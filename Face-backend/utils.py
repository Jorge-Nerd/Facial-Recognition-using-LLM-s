import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from tqdm import tqdm  # Para barra de progresso
import faiss
import sqlite3
import datetime



# Diretórios
known_faces_dir = './known_faces'
os.makedirs(known_faces_dir, exist_ok=True)  # Criar diretório se não existir

#Inicializa Faiss index
dim=512
index_known=faiss.IndexFlatL2(dim)
index_unknown=faiss.IndexFlatL2(dim)



# Verificar disponibilidade de GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Carregue o modelo FaceNet pré-treinado
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

def get_embedding(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Não foi possível ler a imagem: {image_path}")
            return None
            
        img = cv2.resize(img, (160, 160))  # Redimensione a imagem
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter BGR para RGB
        img = img.astype('float32') / 255.0  # Normalização da imagem
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # Tensor no device correto
        
        with torch.no_grad():
            embedding = resnet(img).cpu().detach().numpy()[0]  # Extrair primeiro elemento e converter para CPU
            
        return embedding
    except Exception as e:
        print(f"Erro ao processar {image_path}: {str(e)}")
        return None
    


# Função para calcular o embedding de uma imagem


def add_known_person(first_name, last_name,age,gender,images_path):
    conn=sqlite3.connect('known.db')
    cursor = conn.cursor()
    embeddings=[]
    for filename in os.listdir(images_path):
        img=os.path.join(images_path,filename)
        emb=get_embedding(img)
        if emb is not None:
            embeddings.append(emb)
    
    if not embeddings:
        print("Face not found")
        return
    
    mean_embedding=np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    
    # Adicionar no FAISS
    faiss_id = index_known.nTotal
    index_known.add(np.expand_dims(mean_embedding, axis=0))  # shape (1, 512)
    now = datetime.datetime.now().isoformat()
    
    cursor.execute('''
                   INSERT INTO persons(first_name, last_name,age,gender,faiss_id, last_seen)
                            VALUES(?,?,?,?,?)
                   ''',(first_name, last_name,age,gender,faiss_id, now))
    
    conn.commit()
    print(f"{first_name} {last_name} adicionado com ID {faiss_id}")
    
def add_unknown_person(age,gender,frame):
    conn=sqlite3.connect('unknown.db')
    cursor = conn.cursor()
    embeddings=[]

    emb=get_embedding(frame)
    if emb is not None:
        embeddings.append(emb)

    if not embeddings:
        print("Face not found")
        return

    mean_embedding=np.mean(np.stack(embeddings), axis=0).astype(np.float32)

    # Adicionar no FAISS
    faiss_id = index_unknown.nTotal
    index_unknown.add(np.expand_dims(mean_embedding, axis=0))  # shape (1, 512)
    now = datetime.datetime.now().isoformat()
    
    cursor.execute("SELECT COUNT(*) FROM strangers")
    count = cursor.fetchone()[0]
    name = f"desconhecido_{count + 1}"

    cursor.execute('''
                    INSERT INTO persons(name,age,gender,faiss_id, last_seen)
                            VALUES(?,?,?,?,?)
                    ''',(name,age,gender,faiss_id, now))

    conn.commit()
    conn.close()
    print(f"{name} adicionado com ID {faiss_id}")
