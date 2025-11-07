# Waste Classification with TensorFlow & MobileNetV2

Ce projet est un modèle de classification d’images de déchets (recyclable, non-recyclable, organique, hasardeux) utilisant TensorFlow et MobileNetV2(Transfert Learning) pour une meilleur performence. Il permet d’entraîner un modèle, de le sauvegarder, de le fine-tuner avec de nouvelles données, et de prédire la classe d’images inédites.

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
