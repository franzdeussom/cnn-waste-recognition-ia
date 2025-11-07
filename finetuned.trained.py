import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(180, 180), 
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=(180, 180),
    batch_size=32
)

# 1. Charger le modèle précédemment entraîné
model = tf.keras.models.load_model("final_waste_trained_model.keras", custom_objects={"preprocess_input": preprocess_input})

# 2. Débloquer une partie du base_model
base_model = model.get_layer("mobilenetv2_1.00_224")   # dépend de ta structure, vérifie avec model.summary()
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# 3. Recompiler AVANT de relancer fit()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. Reprendre l'entraînement
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# 5. Sauvegarder la nouvelle version
model.save("final_finetuned.keras")