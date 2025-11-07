import tensorflow as tf
import matplotlib as plt
from tensorflow.keras.callbacks import EarlyStopping 


#import des donné imag train et val

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

class_names = train_ds.class_names
print(f"class_name: {class_names}")
num_class = len(class_names)

#optimise le chagement

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#creation du modl
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(180,180,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_class)
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#compil
model.compile(
    optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics= ['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

#Entrainer
logs = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop])


model.save("final_waste_trained_model.keras")

#visualistio
acc = logs.history['accuracy']
val_acc = logs.history['val_accuracy']
loss = logs.history['loss']
val_loss = logs.history['val_loss']
epochs_rang = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_rang, acc, label="Train Accuracy")
plt.plot(epochs_rang, val_acc, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Precision')

plt.subplot(1,2,2)
plt.plot(epochs_rang, loss, label="Train loss")
plt.plot(epochs_rang, val_loss, label="Validation loss")
plt.legend(loc="Upper right")
plt.title('Erreur')
plt.show()