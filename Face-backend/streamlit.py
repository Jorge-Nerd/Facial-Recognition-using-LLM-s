# Streamlit app
# streamlit_app.py
import streamlit as st
import websocket
import threading
import json
import base64
import cv2
import numpy as np
from PIL import Image
import io
import time
import pandas as pd

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Sistema de Reconhecimento Facial",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para converter base64 para imagem
def base64_to_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    img_io = io.BytesIO(img_bytes)
    img = Image.open(img_io)
    return img

# Inicialização de variáveis de estado
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'ws' not in st.session_state:
    st.session_state.ws = None
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'faces' not in st.session_state:
    st.session_state.faces = []
if 'statistics' not in st.session_state:
    st.session_state.statistics = {"known": 0, "new_unknown": 0, "returning_unknown": 0, "total": 0, "unique_people": 0}
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}

# Título principal
st.title("Sistema de Reconhecimento Facial e Recomendação")

# Layout em colunas
col1, col2 = st.columns([3, 2])

# Coluna 1: Vídeo e estatísticas
with col1:
    # Área de vídeo
    video_placeholder = st.empty()
    
    # Estatísticas
    st.subheader("Estatísticas da Sessão")
    stats_cols = st.columns(4)
    stats_placeholder = [col.empty() for col in stats_cols]
    
    # Pessoas detectadas na sessão atual
    st.subheader("Pessoas Detectadas")
    faces_container = st.empty()

# Coluna 2: Recomendações e Chat
with col2:
    # Recomendações
    st.subheader("Recomendações de Produtos")
    recommendations_container = st.empty()
    
    # Chat
    st.subheader("Chat Interativo")
    chat_container = st.container()
    
    # Área de entrada de mensagem
    message_input = st.text_input("Digite uma mensagem:", key="message_input")
    send_button = st.button("Enviar")

# Sidebar para configurações
with st.sidebar:
    st.header("Configurações")
    
    # Conexão ao WebSocket
    websocket_url = st.text_input("URL do WebSocket", "ws://localhost:8765")
    connect_button = st.button("Conectar", key="connect")
    disconnect_button = st.button("Desconectar", key="disconnect")
    
    # Status da conexão
    connection_status = st.empty()
    
    # Opções de configuração
    st.subheader("Opções")
    show_bounding_boxes = st.checkbox("Mostrar caixas delimitadoras", value=True)
    show_names = st.checkbox("Mostrar nomes", value=True)
    recognition_threshold = st.slider("Limiar de reconhecimento", 0.0, 1.0, 0.6, 0.01)
    
    # Informações do sistema
    st.subheader("Informações do Sistema")
    st.info("""
    Este sistema conecta-se a um servidor de reconhecimento facial via WebSocket.
    O servidor processa o vídeo, identifica pessoas e envia recomendações.
    """)
    
    # Botão para limpar estatísticas
    if st.button("Limpar estatísticas"):
        st.session_state.statistics = {"known": 0, "new_unknown": 0, "returning_unknown": 0, "total": 0, "unique_people": 0}
        st.session_state.faces = []
        st.session_state.chat_messages = []
        st.session_state.recommendations = {}

# Função para processar mensagens do WebSocket
def on_message(ws, message):
    try:
        data = json.loads(message)
        
        # Atualizar frame do vídeo
        if "frame" in data:
            st.session_state.frame = base64_to_image(data["frame"])
        
        # Atualizar rostos detectados
        if "faces" in data:
            st.session_state.faces = data["faces"]
        
        # Atualizar estatísticas
        if "statistics" in data:
            st.session_state.statistics = data["statistics"]
        
        # Atualizar recomendações
        if "recommendations" in data:
            st.session_state.recommendations = data["recommendations"]
        
        # Adicionar mensagens de chat
        if "chat_message" in data:
            st.session_state.chat_messages.append({
                "sender": "Sistema",
                "message": data["chat_message"],
                "timestamp": time.strftime("%H:%M:%S")
            })
    
    except Exception as e:
        st.error(f"Erro ao processar mensagem: {e}")

def on_error(ws, error):
    st.error(f"Erro na conexão WebSocket: {error}")
    st.session_state.connected = False

def on_close(ws, close_status_code, close_msg):
    st.session_state.connected = False
    connection_status.warning("Desconectado do servidor")

def on_open(ws):
    st.session_state.connected = True
    connection_status.success("Conectado ao servidor")
    
    # Enviar configurações iniciais
    settings = {
        "recognition_threshold": recognition_threshold,
        "show_bounding_boxes": show_bounding_boxes,
        "show_names": show_names
    }
    ws.send(json.dumps({"settings": settings}))

# Função para conectar ao WebSocket
def connect_to_websocket():
    if st.session_state.connected:
        return
    
    try:
        ws = websocket.WebSocketApp(
            websocket_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        st.session_state.ws = ws
    except Exception as e:
        st.error(f"Erro ao conectar: {e}")

# Função para desconectar do WebSocket
def disconnect_from_websocket():
    if st.session_state.connected and st.session_state.ws:
        st.session_state.ws.close()
        st.session_state.connected = False
        connection_status.warning("Desconectado do servidor")

# Função para enviar mensagem de chat
def send_chat_message():
    if st.session_state.connected and st.session_state.ws and message_input:
        message_data = {
            "chat_message": message_input
        }
        st.session_state.ws.send(json.dumps(message_data))
        
        # Adicionar mensagem ao chat local
        st.session_state.chat_messages.append({
            "sender": "Você",
            "message": message_input,
            "timestamp": time.strftime("%H:%M:%S")
        })
        
        # Limpar campo de entrada
        st.session_state.message_input = ""

# Atualização contínua da interface
def update_ui():
    # Atualizar vídeo
    if st.session_state.frame is not None:
        video_placeholder.image(st.session_state.frame, caption="Transmissão de vídeo", use_column_width=True)
    else:
        video_placeholder.info("Aguardando transmissão de vídeo...")
    
    # Atualizar estatísticas
    stats = st.session_state.statistics
    stats_placeholder[0].metric("Pessoas conhecidas", stats["known"])
    stats_placeholder[1].metric("Novos desconhecidos", stats["new_unknown"])
    stats_placeholder[2].metric("Desconhecidos retornando", stats["returning_unknown"])
    stats_placeholder[3].metric("Total único", stats["unique_people"])
    
    # Atualizar faces detectadas
    if st.session_state.faces:
        faces_df = pd.DataFrame(st.session_state.faces)
        faces_container.dataframe(faces_df, use_container_width=True)
    else:
        faces_container.info("Nenhuma pessoa detectada")
    
    # Atualizar recomendações
    if st.session_state.recommendations:
        recommendations_html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for person_id, recommendations in st.session_state.recommendations.items():
            person_name = next((face["name"] for face in st.session_state.faces if face["id"] == person_id), "Desconhecido")
            recommendations_html += f"<h4>{person_name} (ID: {person_id})</h4><ul>"
            for rec in recommendations:
                recommendations_html += f"<li><b>{rec['product_name']}</b> - {rec['description']}<br>Confiança: {rec['confidence']:.2f}</li>"
            recommendations_html += "</ul>"
        recommendations_html += "</div>"
        recommendations_container.markdown(recommendations_html, unsafe_allow_html=True)
    else:
        recommendations_container.info("Nenhuma recomendação disponível")
    
    # Atualizar chat
    chat_html = "<div style='max-height: 400px; overflow-y: auto;'>"
    for msg in st.session_state.chat_messages:
        sender_style = "color: blue;" if msg["sender"] == "Você" else "color: green;"
        chat_html += f"<p><span style='{sender_style}'><b>{msg['sender']}</b> ({msg['timestamp']}):</span><br>{msg['message']}</p>"
    chat_html += "</div>"
    
    with chat_container:
        st.markdown(chat_html, unsafe_allow_html=True)

# Processar botões de ação
if connect_button:
    connect_to_websocket()

if disconnect_button:
    disconnect_from_websocket()

if send_button or (message_input and st.session_state.message_input != message_input):
    send_chat_message()

# Atualizar UI
update_ui()

# Conteúdo adicional - explicação sobre o sistema
st.markdown("---")
with st.expander("Sobre o Sistema de Reconhecimento Facial"):
    st.markdown("""
    ### Como funciona o Sistema
    
    Este sistema utiliza uma arquitetura cliente-servidor onde:
    
    1. **Servidor Backend**: Processa vídeo em tempo real com modelos de reconhecimento facial
    2. **Frontend Streamlit**: Exibe os resultados e permite interação com o sistema
    
    ### Funcionalidades
    
    - **Reconhecimento facial** em tempo real
    - **Identificação** de pessoas conhecidas e desconhecidas
    - **Recomendações personalizadas** baseadas em histórico e perfil
    - **Chat interativo** para consultas e feedback
    
    ### Privacidade e Segurança
    
    Os dados faciais são processados com padrões rigorosos de privacidade. Nenhuma imagem é armazenada permanentemente sem consentimento explícito.
    """)

# Timer para atualização automática (opcional)
# No ambiente Streamlit, a página é recarregada a cada interação, então não é necessário um timer explícito