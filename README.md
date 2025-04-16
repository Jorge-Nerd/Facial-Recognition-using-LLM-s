# Tese

# Sistema de Reconhecimento Facial Inteligente com Cadastro em Tempo Real

Este projeto é um sistema completo de reconhecimento facial com suporte a:

📸 Cadastro dinâmico de pessoas conhecidas com 3 fotos

🤖 Predição de idade e gênero

🧠 Reconhecimento de rostos conhecidos e visitantes recorrentes

🕵️ Detecção e cadastro automático de visitantes novos

⚡ Funciona em tempo real, via web app leve com Streamlit

#✨ Funcionalidades
Detecção facial: YOLOv8n-face (modelo leve e rápido)

Reconhecimento facial: FaceNet (InceptionResNetV1)

Predição de idade e gênero: modelo PyTorch customizado

Indexação facial: FAISS (Facebook AI Similarity Search)

Base de dados: SQLite para metadados (nome, idade, gênero)

Interface gráfica: Streamlit (leve e 100% Python)

#📁 Estrutura de Pastas
bash
Copiar
Editar
📦 Projeto
├── known_faces/             # Armazena imagens das pessoas cadastradas
├── faiss_known.index        # Índice FAISS com embeddings de pessoas conhecidas
├── faiss_unknown.index      # Índice FAISS com embeddings de pessoas desconhecidas
├── known.db                 # Banco de dados SQLite (conhecidos)
├── unknown.db               # Banco de dados SQLite (desconhecidos)
├── gender_age_model_new.pth # Modelo PyTorch de predição de idade e gênero
├── yolov8n-face.pt          # Modelo YOLOv8 para detecção facial
├── utils_models.py          # Modelo de idade/gênero
├── utils.py                 # Funções auxiliares (embeddings, reconhecimento, etc)
├── db_manager.py            # Funções de criação e manipulação dos bancos
└── app.py                   # Aplicação principal com Streamlit

#⚙️ Requisitos
Python 3.8+

PyTorch

OpenCV

Streamlit

FAISS

Pillow

torchvision

facenet-pytorch

ultralytics (YOLO)

SQLite3 (incluso no Python)

Instalação via requirements.txt:

bash
Copiar
Editar
pip install -r requirements.txt
#🚀 Como Executar
Certifica-te que tens a webcam funcionando.

Coloca os arquivos dos modelos (yolov8n-face.pt, gender_age_model_new.pth) na raiz do projeto.

Roda o app com Streamlit:

bash
Copiar
Editar
streamlit run app.py
Acede no navegador (geralmente abre automaticamente):

arduino
Copiar
Editar
http://localhost:8501
🧪 Como Testar
Na seção "Adicionar nova pessoa", preenche nome, idade e gênero, e faz upload de 3 fotos.

Ativa a câmera em "Câmera em tempo real com reconhecimento".

O sistema vai:

Reconhecer pessoas cadastradas com nome e dados.

Detectar visitantes recorrentes (desconhecidos já vistos).

Cadastrar novos visitantes automaticamente com ID incremental.

#🛡️ Segurança & Privacidade
Todo o processamento é feito localmente, sem upload para a internet.

Ideal para sistemas offline e com restrições de privacidade.

#📌 Possíveis Melhorias Futuras
Exportar dados para CSV/Excel.

Dashboard de estatísticas com Streamlit.

Módulo de notificação (email ou Telegram).

Otimização com TensorRT para maior velocidade.

👨‍💻 Autor
Teu Nome
Engenheiro de Dados & IA
📍 Lisboa, Portugal
🔗 LinkedIn | GitHub
