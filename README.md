# Skin Disease Classifier using Vision Transformer (ViT) 🩺

A deep learning web application that classifies dermoscopic skin lesion images into seven categories using a Vision Transformer (ViT) model fine-tuned on the HAM10000 dataset.

## 🚀 Live Demo

**[Try the Skin Disease Classifier](https://skindiseaseclassifier-ia5j.onrender.com)**

Upload a dermoscopic skin lesion image to view the model's top 3 predictions and their confidence scores.

> **Disclaimer:** This project is intended for educational and research purposes only. Predictions are not medical diagnoses and should not replace evaluation by a qualified healthcare professional.

## ✨ Features

- Classifies dermoscopic images into seven lesion categories.
- Displays the top 3 predictions with confidence scores.
- Uses a Vision Transformer model implemented with PyTorch and timm.
- Provides an interactive web interface built with Gradio.
- Supports CPU-based inference.
- Deployed on Render with a publicly accessible web interface.

## 🧠 Model

- **Architecture:** `vit_tiny_patch16_224`
- **Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Framework:** PyTorch
- **Model library:** timm
- **Checkpoint:** `vit_skin_disease.pth`

### Supported Categories

| Label | Skin Lesion Category |
|---|---|
| `akiec` | Actinic keratoses |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `nv` | Melanocytic nevi |
| `mel` | Melanoma |
| `vasc` | Vascular lesions |

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, torchvision, timm
- **Web Interface:** Gradio
- **Image Processing:** Pillow
- **Additional Libraries:** NumPy, scikit-learn, Matplotlib
- **Deployment:** Render

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/123cs0178/SkinDiseaseClassifier.git
cd SkinDiseaseClassifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

Open the local URL displayed in the terminal.

Ensure that the trained checkpoint `vit_skin_disease.pth` is present in the project directory.

## ☁️ Deployment

The application is deployed as a Python web service on Render.

- **Python version:** 3.12.11
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `python app.py`
- **Inference:** CPU

The free Render service may spin down after inactivity, so the first request after a period of inactivity may take longer to respond.

## 📁 Project Structure

```text
SkinDiseaseClassifier/
├── app.py
├── requirements.txt
├── vit_skin_disease.pth
├── SkinDiseaseClassifier (3).ipynb
├── README.md
└── LICENSE
```

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
