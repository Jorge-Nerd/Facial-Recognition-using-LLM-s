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
| ![Cadastro](assets/add_person.png) | ![Câmera](assets/realtime_recognition.png) |

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
[git clone https://github.com/seu-usuario/sistema-reconhecimento-facial.git](https://github.com/Jorge-Nerd/Tese.git)
cd Tese
