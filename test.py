import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "final_waste_trained_model.keras" 
TEST_DIR = "dataset/testshuffle"                             
IMG_SIZE = (180, 180)                               
CLASS_NAMES = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable'] 

print(f"Chargement du modèle depuis {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"preprocess_input": preprocess_input})
print(" Modèle chargé avec succès....\n")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    # img_array =img_array 

    predictions = model.predict(img_array, verbose=0)

    pred_class = np.argmax(predictions[0]) # class d'appartenance predite
    prob = np.max(tf.nn.softmax(predictions[0])) # probabilité de confianc
    
    return CLASS_NAMES[pred_class], prob

if not os.path.exists(TEST_DIR):
    print(f"⚠️ Le dossier '{TEST_DIR}' est introuvable.")
else:
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print(f"⚠️ Aucun fichier image trouvé dans '{TEST_DIR}'.")
    else:
        print(f"Prédictions pour {len(files)} image(s) :\n")
        for file in files:
            path = os.path.join(TEST_DIR, file)
            label, conf = predict_image(path)
            print(f"{file}: {label} à ({conf*100:.2f}% de precision)")
