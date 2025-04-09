# db_manager.py
import sqlite3
import os
import numpy as np
import pickle
import json
from datetime import datetime
import faiss

class DBManager:
    def __init__(self, db_path='./data/face_recognition.db'):
        # Criar diretório para o banco de dados se não existir
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
        # Inicializar índice FAISS
        self.faiss_index = None
        self.faiss_ids = []
        self.initialize_faiss()
    
    def create_tables(self):
        """Criar tabelas necessárias no banco de dados"""
        cursor = self.conn.cursor()
        
        # Tabela de pessoas conhecidas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabela de detecções
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            face_id TEXT,
            embedding BLOB NOT NULL,
            age REAL,
            gender TEXT,
            person_type TEXT,
            person_name TEXT,
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabela de sessões
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            unique_people INTEGER,
            known_count INTEGER,
            new_unknown_count INTEGER,
            returning_unknown_count INTEGER,
            total_detections INTEGER
        )
        ''')
        
        # Tabela de recomendações
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY,
            detection_id INTEGER,
            products TEXT,
            FOREIGN KEY (detection_id) REFERENCES detections (id)
        )
        ''')
        
        self.conn.commit()
    
    def initialize_faiss(self):
        """Inicializar índice FAISS com embeddings de pessoas conhecidas"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM known_faces")
        rows = cursor.fetchall()
        
        if rows:
            # Extrair IDs e embeddings
            self.faiss_ids = [row[0] for row in rows]
            embeddings = [pickle.loads(row[1]) for row in rows]
            
            # Criar e preencher índice FAISS
            dimension = embeddings[0].shape[0]  # Dimensão do embedding (geralmente 512 para FaceNet)
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np.array(embeddings).astype('float32'))
            
            print(f"Índice FAISS inicializado com {len(self.faiss_ids)} embeddings")
        else:
            # Criar índice vazio se não houver pessoas conhecidas
            self.faiss_index = faiss.IndexFlatL2(512)  # Assumindo dimensão 512 para FaceNet
            print("Índice FAISS inicializado vazio")
    
    def add_known_face(self, name, embedding, image_path=None):
        """Adicionar uma pessoa conhecida ao banco de dados"""
        cursor = self.conn.cursor()
        
        # Verificar se já existe
        cursor.execute("SELECT id FROM known_faces WHERE name = ?", (name,))
        existing = cursor.fetchone()
        
        if existing:
            # Atualizar embedding existente
            cursor.execute(
                "UPDATE known_faces SET embedding = ?, image_path = ? WHERE id = ?",
                (pickle.dumps(embedding), image_path, existing[0])
            )
            face_id = existing[0]
            print(f"Atualizado embedding para {name} (ID: {face_id})")
        else:
            # Inserir novo
            cursor.execute(
                "INSERT INTO known_faces (name, embedding, image_path) VALUES (?, ?, ?)",
                (name, pickle.dumps(embedding), image_path)
            )
            face_id = cursor.lastrowid
            print(f"Adicionada nova pessoa conhecida: {name} (ID: {face_id})")
        
        self.conn.commit()
        
        # Atualizar índice FAISS
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatL2(embedding.shape[0])
        
        if existing:
            # Remover e adicionar novamente para atualizar
            idx = self.faiss_ids.index(existing[0])
            self.faiss_ids.pop(idx)
            self.faiss_index.remove_ids(np.array([idx]))
        
        # Adicionar ao índice FAISS
        self.faiss_index.add(np.array([embedding]).astype('float32'))
        self.faiss_ids.append(face_id)
        
        return face_id
    
    def identify_face(self, embedding, threshold=0.8):
        """Identificar rosto usando FAISS"""
        if not self.faiss_ids:  # Se não houver rostos conhecidos
            return None, float('inf')
        
        # Pesquisar no índice FAISS
        distances, indices = self.faiss_index.search(np.array([embedding]).astype('float32'), 1)
        
        closest_distance = distances[0][0]
        if closest_distance < threshold:
            closest_person_id = self.faiss_ids[indices[0][0]]
            
            # Obter nome da pessoa
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM known_faces WHERE id = ?", (closest_person_id,))
            result = cursor.fetchone()
            
            if result:
                return result[0], closest_distance
        
        return None, closest_distance
    
    def add_detection(self, face_id, embedding, age, gender, person_type, person_name):
        """Registrar uma detecção no banco de dados"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """INSERT INTO detections 
               (face_id, embedding, age, gender, person_type, person_name) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (face_id, pickle.dumps(embedding), age, gender, person_type, person_name)
        )
        
        detection_id = cursor.lastrowid
        self.conn.commit()
        
        return detection_id
    
    def add_recommendations(self, detection_id, products):
        """Adicionar recomendações para uma detecção"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            "INSERT INTO recommendations (detection_id, products) VALUES (?, ?)",
            (detection_id, json.dumps(products))
        )
        
        self.conn.commit()
    
    def start_session(self):
        """Iniciar uma nova sessão de detecção"""
        cursor = self.conn.cursor()
        
        start_time = datetime.now()
        cursor.execute(
            "INSERT INTO sessions (start_time, unique_people, known_count, new_unknown_count, returning_unknown_count, total_detections) VALUES (?, 0, 0, 0, 0, 0)",
            (start_time,)
        )
        
        session_id = cursor.lastrowid
        self.conn.commit()
        
        return session_id
    
    def update_session(self, session_id, stats):
        """Atualizar estatísticas da sessão"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """UPDATE sessions SET 
               unique_people = ?, 
               known_count = ?, 
               new_unknown_count = ?, 
               returning_unknown_count = ?, 
               total_detections = ?,
               end_time = ?
               WHERE id = ?""",
            (
                stats["unique_people"],
                stats["known"],
                stats["new_unknown"],
                stats["returning_unknown"],
                stats["total"],
                datetime.now(),
                session_id
            )
        )
        
        self.conn.commit()
    
    def get_detection_history(self, face_id=None, limit=10):
        """Obter histórico de detecções"""
        cursor = self.conn.cursor()
        
        if face_id:
            cursor.execute(
                """SELECT id, detection_time, age, gender, person_type, person_name 
                   FROM detections WHERE face_id = ? 
                   ORDER BY detection_time DESC LIMIT ?""",
                (face_id, limit)
            )
        else:
            cursor.execute(
                """SELECT id, detection_time, age, gender, person_type, person_name, face_id 
                   FROM detections ORDER BY detection_time DESC LIMIT ?""",
                (limit,)
            )
        
        results = cursor.fetchall()
        
        # Formatar resultados
        history = []
        for row in results:
            if face_id:
                history.append({
                    "id": row[0],
                    "detection_time": row[1],
                    "age": row[2],
                    "gender": row[3],
                    "person_type": row[4],
                    "person_name": row[5]
                })
            else:
                history.append({
                    "id": row[0],
                    "detection_time": row[1],
                    "age": row[2],
                    "gender": row[3],
                    "person_type": row[4],
                    "person_name": row[5],
                    "face_id": row[6]
                })
        
        return history
    
    def get_known_faces(self):
        """Obter lista de pessoas conhecidas"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, image_path FROM known_faces")
        
        results = cursor.fetchall()
        known_faces = []
        
        for row in results:
            known_faces.append({
                "id": row[0],
                "name": row[1],
                "image_path": row[2]
            })
        
        return known_faces
    
    def close(self):
        """Fechar conexão com o banco de dados"""
        if self.conn:
            self.conn.close()