import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ----------- 1. Dizinleri Tanımla -----------
train_dir = os.path.join("/Users/veysel/Desktop/DerinOgrenme/Brain_Tumor_Dataset/Training")
test_dir = os.path.join("/Users/veysel/Desktop/DerinOgrenme/Brain_Tumor_Dataset/Testing")
img_size = 150
batch_size = 32
epochs = 40  # Optimize modelde genelde daha fazla epoch kullanabilirsin

# ----------- 2. Veri ve Augmentation Pipeline -----------
def get_data_generators(train_dir, test_dir, img_size=150, batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.1,
        brightness_range=[0.8,1.2]
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    class_names = list(train_gen.class_indices.keys())
    return train_gen, test_gen, class_names

# ----------- 3. Optimize CNN Modeli -----------
def create_cnn_model(input_shape=(150, 150, 3), num_classes=4):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ----------- 4. Eğitim Pipeline -----------
train_gen, test_gen, class_names = get_data_generators(train_dir, test_dir, img_size, batch_size)
num_classes = len(class_names)

print(f"Sınıf isimleri: {class_names}")
print(f"Toplam sınıf: {num_classes}")

model = create_cnn_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------- 5. Callbacks (EarlyStopping & LR Reduce) -----------

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# ----------- 6. Modeli Eğit -----------
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=test_gen,
    callbacks=callbacks
)

# ----------- 7. Modeli Kaydet -----------
model.save("brain_tumor_optimized_cnn_model.keras")

# ----------- 8. Eğitim Sonuçlarını Görselleştir -----------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title('Başarı Oranları')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
