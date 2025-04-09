# face_processor.py
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import os
from utils_models import GenderAgeModel
from db_manager import DBManager
from ultralytics import YOLO

class FaceProcessor:
    def __init__(self):
        # Definir dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Carregar modelos
        self.load_models()
        
        # Inicializar DB Manager
        self.db = DBManager()
        
        # Transformações para imagens
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Estatísticas da sessão atual
        self.session_id = self.db.start_session()
        self.current_session_people = set()
        self.session_statistics = {
            "known": 0,
            "new_unknown": 0,
            "returning_unknown": 0,
            "total": 0,
            "unique_people": 0
        }
        
        # Histórico recente de detecções para controle de duplicatas
        self.recent_detections = {}
    
    def load_models(self):
        """Carregar todos os modelos necessários"""
        # Modelo FaceNet para embeddings faciais
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Modelo YOLO para detecção facial
        self.face_detector = YOLO('./yolov8n-face.pt')
        
        # Modelo de gênero e idade
        self.gender_age_model = GenderAgeModel()
        self.gender_age_model.load_state_dict(torch.load('./gender_age_model_new.pth', map_location=self.device))
        self.gender_age_model.to(self.device)
        self.gender_age_model.eval()
        
        print("Todos os modelos carregados com sucesso")
    
    def get_face_embedding(self, face_img):
        """Extrair embedding de um rosto detectado"""
        # Converter BGR para RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para 160x160 para FaceNet
        face_img_resized = cv2.resize(face_img_rgb, (160, 160))
        
        # Normalizar e converter para tensor
        face_tensor = torch.from_numpy(face_img_resized).float().permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor / 255.0  # Normalizar para [0, 1]
        face_tensor = face_tensor.to(self.device)
        
        # Extrair embedding
        with torch.no_grad():
            embedding = self.resnet(face_tensor).cpu().numpy().flatten()
        
        return embedding
    
    def predict_gender_age(self, face_img):
        """Prever gênero e idade de um rosto"""
        # Converter BGR para RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Converter para PIL e aplicar transformações
        pil_img = Image.fromarray(face_img_rgb)
        face_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Prever gênero e idade
        with torch.no_grad():
            age_pred, gender_pred = self.gender_age_model(face_tensor)
            
            # Converter para valores interpretáveis
            age = age_pred.item() * 100  # Assumindo que foi normalizado para 0-1
            gender = "Masculino" if gender_pred.item() > 0.5 else "Feminino"
            
        return age, gender
    
    def is_duplicate_detection(self, face_id, embedding):
        """Verificar se esta é uma detecção duplicada recente"""
        # Calcular hash do embedding para comparação rápida
        embedding_hash = hash(embedding.tobytes())
        
        # Verificar se o rosto já foi detectado recentemente
        if face_id in self.recent_detections:
            if embedding_hash == self.recent_detections[face_id]["hash"]:
                # Atualizar timestamp e retornar True (é duplicado)
                self.recent_detections[face_id]["timestamp"] = time.time()
                return True
        
        # Não é duplicado, adicionar ao histórico recente
        self.recent_detections[face_id] = {
            "hash": embedding_hash,
            "timestamp": time.time()
        }
        
        # Limpar detecções antigas (mais de 5 segundos)
        current_time = time.time()
        for face_id in list(self.recent_detections.keys()):
            if current_time - self.recent_detections[face_id]["timestamp"] > 5:
                del self.recent_detections[face_id]
        
        return False
    
    def process_face(self, face_img, face_id):
        """Processar um rosto detectado"""
        # Extrair embedding
        embedding = self.get_face_embedding(face_img)
        
        # Verificar se é uma detecção duplicada recente
        if self.is_duplicate_detection(face_id, embedding):
            return None
        
        # Identificar rosto
        person_name, confidence = self.db.identify_face(embedding)
        
        # Prever gênero e idade
        age, gender = self.predict_gender_age(face_img)
        
        # Determinar tipo de pessoa
        if person_name:
            person_type = "known"
            self.session_statistics["known"] += 1
        else:
            # Verificar se este rosto foi visto antes nesta sessão
            is_new = face_id not in self.current_session_people
            
            if is_new:
                person_type = "new_unknown"
                self.session_statistics["new_unknown"] += 1
                self.current_session_people.add(face_id)  # Adicionar à lista de pessoas vistas
            else:
                person_type = "returning_unknown"
                self.session_statistics["returning_unknown"] += 1
        
        # Adicionar ao contador total
        self.session_statistics["total"] += 1
        self.session_statistics["unique_people"] = len(self.current_session_people)
        
        # Salvar detecção no banco de dados
        detection_id = self.db.add_detection(face_id, embedding, age, gender, person_type, person_name)
        
        # Gerar recomendações de produtos
        recommendations = self.get_product_recommendations(age, gender)
        self.db.add_recommendations(detection_id, recommendations)
        
        # Atualizar estatísticas da sessão no banco
        self.db.update_session(self.session_id, self.session_statistics)
        
        # Criar resultado final
        result = {
            "detection_id": detection_id,
            "person_name": person_name,
            "confidence": 1.0 - confidence / 2.0 if person_name else 0,  # Normalizar para 0-1
            "person_type": person_type,
            "age": age,
            "gender": gender,
            "recommendations": recommendations
        }
        
        return result
    
    def get_product_recommendations(self, age, gender):
        """Gerar recomendações baseadas em idade e gênero"""
        # Esta função seria substituída pela integração com LLaMA
        # Por enquanto, usar uma lógica simples baseada em regras
        
        recommendations = []
        
        if gender == "Masculino":
            if age < 18:
                recommendations = ["Videogames", "Camisetas casuais", "Tênis esportivos"]
            elif age < 30:
                recommendations = ["Smartphones", "Roupas fitness", "Relógios"]
            elif age < 50:
                recommendations = ["Notebooks", "Camisas sociais", "Perfumes"]
            else:
                recommendations = ["Relógios clássicos", "Óculos de leitura", "Produtos de saúde"]
        else:  # Feminino
            if age < 18:
                recommendations = ["Acessórios para cabelo", "Produtos de beleza leve", "Tênis coloridos"]
            elif age < 30:
                recommendations = ["Bolsas", "Maquiagem", "Produtos para cabelo"]
            elif age < 50:
                recommendations = ["Cosméticos anti-idade", "Acessórios", "Roupas de escritório"]
            else:
                recommendations = ["Cremes anti-idade", "Joias", "Produtos de bem-estar"]
        
        return recommendations
    
    def finalize_session(self):
        """Finalizar a sessão atual"""
        # Atualizar estatísticas da sessão no banco
        self.db.update_session(self.session_id, self.session_statistics)
        print(f"Sessão {self.session_id} finalizada com sucesso")
        
        # Fechar conexão com o banco
        self.db.close()