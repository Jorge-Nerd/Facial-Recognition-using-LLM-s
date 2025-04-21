# Tese

# 🧠 Sistema Inteligente de Reconhecimento Facial com Cadastro em Tempo Real

Este é um sistema completo de reconhecimento facial com predição de idade e gênero, integração com banco de dados e FAISS para busca de vetores faciais. Ele permite o **cadastro em tempo real de novas pessoas**, reconhecimento de **rostos conhecidos** e **visitantes desconhecidos recorrentes**, com uma interface interativa em **Streamlit**.

---

## ✨ Funcionalidades

- 🎥 Reconhecimento facial em tempo real com webcam
- 🧑‍💼 Cadastro de novas pessoas com nome, idade e gênero
- 🔍 Identificação de pessoas conhecidas via FAISS
- 🤖 Predição de idade e gênero com modelo treinado
- 📁 Armazenamento e distinção de rostos desconhecidos (recorrentes ou novos)
- 🗃️ Banco de dados SQLite para metadados
- ⚡ Suporte a GPU (CUDA) ou CPU

---

## 🖼️ Interface

| Cadastro de Pessoas | Reconhecimento em Tempo Real |
|---------------------|-------------------------------|
| ![cadastro](https://github.com/user-attachments/assets/7462df01-7ac6-4203-8b65-7e8366f3cae8)
erson.png) | ![Câmera](assets/realtime_recognition.png) |

---

## 🛠️ Tecnologias Utilizadas

- [Python 3.8+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [FaceNet (InceptionResnetV1)](https://github.com/timesler/facenet-pytorch)
- [PyTorch](https://pytorch.org/)
- SQLite

---

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/Jorge-Nerd/Tese.git
cd Tese/Face-backend
```
2. Crie um ambiente virtual e ative:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute a aplicação:

```bash
streamlit run app.py
```

## Arquitetura do Sistema

                          +---------------------+
                          |     Streamlit UI    |
                          +----------+----------+
                                     |
                                     v
               +--------------------+--------------------+
               |    Fast Face Detection (YOLOv8n-face)   |
               +--------------------+--------------------+
                                     |
        +----------------------------+----------------------------+
        |            +---------------v--------------+             |
        |            |   Face Embedding (FaceNet)   |             |
        |            +---------------+--------------+             |
        |                            |                            |
        |                            v                            |
        |      +--------------------+--------------------+        |
        |      |         FAISS Search (Known / Unknown)  |        |
        |      +--------------------+--------------------+        |
        |                            |                            |
        |         +------------------+-----------------+          |
        |         |       SQLite (known.db / unknown.db)         |
        |         +----------------------------------------------+


##  🗃️ Estrutura de Diretórios

├── app.py
├── utils.py
├── utils_models.py
├── db_manager.py
├── gender_age_model_new.pth
├── yolov8n-face.pt
├── known_faces/
├── faiss_known.index
├── faiss_unknown.index
├── known.db
├── unknown.db
├── requirements.txt
└── README.md


##  🙋‍♂️ Sobre
Este projeto foi desenvolvido como parte de uma tese de mestrado, com foco em reconhecimento facial inteligente aplicado em contextos como DOOH (Digital Out Of Home) e marketing personalizado com base em idade e gênero preditos.
