import cv2
import asyncio
import uvicorn
import base64
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from face_processor import FaceProcessor
import time
import uuid

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gerenciador de conexões WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_data(self, websocket: WebSocket, data: str):
        await websocket.send_text(data)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Inicializar processador facial
    face_processor = FaceProcessor()
    
    # Iniciar captura de vídeo
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        await websocket.close(code=1000, reason="Erro ao acessar a câmera")
        return

    try:
        next_face_id = 1
        face_ids = {}  # Mapear bounding boxes para IDs
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar faces com YOLO
            results = face_processor.face_detector(frame)
            
            # Processar cada rosto detectado
            face_data = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extrair rosto
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:  # Pular se o recorte estiver vazio
                        continue
                    
                    # Gerar ou reutilizar ID para este rosto
                    face_bbox_key = f"{x1}_{y1}_{x2}_{y2}"
                    
                    if face_bbox_key not in face_ids:
                        face_ids[face_bbox_key] = f"face_{next_face_id}"
                        next_face_id += 1
                    
                    face_id = face_ids[face_bbox_key]
                    
                    # Processar o rosto
                    try:
                        face_result = face_processor.process_face(face_img, face_id)
                        
                        # Pular se for uma detecção duplicada
                        if face_result is None:
                            continue
                        
                        # Adicionar informações da bounding box
                        face_result["bbox"] = [x1, y1, x2, y2]
                        face_result["face_id"] = face_id
                        
                        face_data.append(face_result)
                        
                        # Desenhar retângulo e informações no frame
                        label = f"{face_result['person_name'] if face_result['person_name'] else 'Desconhecido'}"
                        confidence = face_result['confidence']
                        
                        color = (0, 255, 0) if face_result['person_name'] else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Adicionar informações
                        y_offset = y1 - 10
                        cv2.putText(frame, label, (x1, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        y_offset += 20
                        cv2.putText(frame, f"Idade: {int(face_result['age'])}", (x1, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        y_offset += 20
                        cv2.putText(frame, f"Gênero: {face_result['gender']}", (x1, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    except Exception as e:
                        print(f"Erro ao processar rosto: {e}")
            
            # Converter frame para formato base64 para enviar via WebSocket
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Criar dados para enviar
            data = {
                "frame": img_base64,
                "faces": face_data,
                "statistics": face_processor.session_statistics
            }
            
            # Enviar dados via WebSocket
            await manager.send_data(websocket, json.dumps(data))
            
            # Pequeno delay para controlar a taxa de frames
            await asyncio.sleep(0.05)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket desconectado")
    except Exception as e:
        print(f"Erro no WebSocket: {e}")
    finally:
        # Finalizar sessão
        face_processor.finalize_session()
        
        # Liberar recursos
        cap.release()