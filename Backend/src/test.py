import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------- Dizin ve Parametreler -----------
test_dir = os.path.join("/Users/veysel/Desktop/DerinOgrenme/Brain_Tumor_Dataset/Testing")
img_size = 150
batch_size = 32

# ----------- Test Veri Jeneratörü (ve class_names) -----------
def get_test_generator(test_dir, img_size=150, batch_size=32):
    test_gen = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False
    )
    class_names = test_gen.class_names
    return test_gen, class_names

test_gen, class_names = get_test_generator(test_dir, img_size, batch_size)

# ----------- Eğitilmiş Modeli Yükle -----------
model = tf.keras.models.load_model("/Users/veysel/Desktop/DerinOgrenme/brain_tumor_optimized_cnn_model.h5")  # veya .keras

# ----------- Tahmin & Analiz -----------
y_true = []
y_pred = []

for images, labels in test_gen:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Karışıklık Matrisi (Confusion Matrix)")
plt.show()

print("Sınıf bazında performans raporu:")
print(classification_report(y_true, y_pred, target_names=class_names))
