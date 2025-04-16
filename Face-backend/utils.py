import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import sqlite3
import datetime
import os
import faiss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dim = 512
index_known = faiss.IndexFlatL2(dim)
index_unknown = faiss.IndexFlatL2(dim)

def get_embedding(image):
    try:
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        if img is None:
            print("Imagem inválida.")
            return None

        img = cv2.resize(img, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(img).cpu().numpy()[0]

        return embedding

    except Exception as e:
        print(f"Erro no embedding: {str(e)}")
        return None


def get_average_embedding(person_dir):
    embeddings = []
    for image_file in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_file)
        if os.path.isfile(image_path):
            embedding = get_embedding(image_path)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                print(f"Embedding inválido: {image_path}")

    if len(embeddings) == 0:
        print("Nenhum embedding válido foi encontrado.")
        return None

    average_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    return average_embedding


def recognize_face(frame, index, conn, threshold=0.8):
    """
    Reconhece um rosto com um frame comparado seu embedding com os embeddings conhecidos.
    
    Args:
        frame: Frame da pessoa a ser detetado em tempo real que queremos reconhecer
        index_known (faiss.Index): Index do faiss que está associado ao embedding dos rostos conhecidos
        conn_known (sqlite3.Connection): Conexão com o bando de dados SQlite de rostos conhecidos
        threshold (float): Limiar de distançia
    
    Returns:
        tuple or None: Se um rosto reconhecido foi identificado como abaixo do limiar retorna uma tupla
        contendo as informacoes(nome completo,idade,genero)
    
    """
    emb = get_embedding(frame)
    if emb is None or index.ntotal == 0:
        return None

    D, I = index.search(np.expand_dims(emb, axis=0), k=1)
    if D[0][0] < threshold:
        faiss_id = int(I[0][0])
        cursor = conn.cursor()
        cursor.execute("SELECT full_name, age, gender FROM persons WHERE faiss_id=?", (faiss_id,))
        res = cursor.fetchone()
        if res:
            full_name, age, gender = res
            now = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE persons SET last_seen=? WHERE faiss_id=?", (now, faiss_id))
            conn.commit()
            return full_name, age, gender, now
    return None

def recognize_unknown_face(frame, index, conn, threshold=0.75):
    """
    Identificar um rosto desconhecido comparando o seu embedding com os embeddings dos rostos deconhecidos guardados no FAISS
    
    Args:
        frame: É o frame(rosto) da pessoa a ser identificada em tempo real
        index_unknown: É o index FAISS contendo os rostos das pessoas desconhecidaos
        conn_unknown (sqlite3.Connection): Conexão com o banco de dados SQlite ods rostos desconhecidos
        threshold (float): Limiar de distância para considerar uma correspondência.

    Returns:
        tuple or None: Se um rosto desconhecido similar for encontrado abaixo do limiar, retorna uma tupla
                       contendo (nome, idade, gênero, faiss_id). Retorna None caso contrário.
    """ 

    emb = get_embedding(frame)
    if emb is None or index.ntotal == 0:
        return None

    D, I = index.search(np.expand_dims(emb, axis=0), k=1)
    if D[0][0] < threshold:
        faiss_id = int(I[0][0])
        cursor = conn.cursor()
        cursor.execute("SELECT name, age, gender FROM strangers WHERE faiss_id=?", (faiss_id,))
        res = cursor.fetchone()
        if res:
            name, age, gender = res
            now = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE strangers SET last_seen=?, total_seen=total_seen+1 WHERE faiss_id=?", (now, faiss_id))
            conn.commit()
            return name, age, gender, now
    return None

def add_person(full_name, age, gender, person_dir, index_known, conn_known):
    cursor = conn_known.cursor()

    average_embedding = get_average_embedding(person_dir)
    if average_embedding is None:
        return None

    faiss_id = index_known.ntotal
    index_known.add(np.expand_dims(average_embedding, axis=0).astype(np.float32))
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO persons (faiss_id,full_name, age, gender, last_seen)
        VALUES (?, ?, ?, ?,?)
    ''', ( faiss_id,full_name, age, gender,now))

    conn_known.commit()
    return {
        "faiss_id": faiss_id,
        "name": full_name,
        "age": age,
        "gender": gender,
        "path": person_dir
    }

def add_unknown_person(age, gender, frame, index_unknown, conn_unknown):
    cursor = conn_unknown.cursor()

    emb = get_embedding(frame)
    if emb is None:
        return None

    faiss_id = index_unknown.ntotal
    index_unknown.add(np.expand_dims(emb, axis=0))
    now = datetime.datetime.now().isoformat()

    cursor.execute("SELECT COUNT(*) FROM strangers")
    count = cursor.fetchone()[0]
    name = f"Visitante_{count+1}"

    cursor.execute('''
        INSERT INTO strangers (name, age, gender, faiss_id, first_seen, last_seen, total_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, age, gender, faiss_id, now, now, 1))

    conn_unknown.commit()
    return name

