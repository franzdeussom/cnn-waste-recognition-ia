# Waste Classification with TensorFlow & MobileNetV2

This project is a waste image classification model (recyclable, non-recyclable, organic, hazardous) using TensorFlow and MobileNetV2, and a Transfer Learning technique for improved model performance. It allows users to train a model, save it, fine-tune it with new data, and predict the class of a new image.

---

## 1. Prérequis

- Python 3.10+ recommandé
- pip installé

---

## 2. Création de l’environnement virtuel

```bash

cd mon_projet_dechets

python3 -m venv venv

# Activer l'environnement virtuelle linux windos
source venv/bin/activate
#ou 
venv\Scripts\activate

# Mettre à jour pip
pip install --upgrade pip
