import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Константы
IMG_SIZE = (227, 227)
BATCH_SIZE = 64
CLASS_NAMES = ["Not Tired", "Tired"]
MODEL_PATH = "best_model.keras"
OPTIMAL_THRESHOLD = 0.5

# ====================
# Загрузка датасета
# ====================
def get_dataset(path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='binary',
        shuffle=False
    )

# ====================
# Визуализация ошибок
# ====================
def show_misclassified_examples(generator, predictions, labels, threshold=0.5, max_examples=8):
    predicted_labels = (predictions > threshold).astype(int)
    misclassified_idxs = np.where(predicted_labels != labels)[0]
    print(f"\n🔎 Найдено {len(misclassified_idxs)} ошибок классификации.")

    if len(misclassified_idxs) == 0:
        print("✅ Ошибок нет — модель справилась отлично!")
        return

    num_to_show = min(max_examples, len(misclassified_idxs))
    plt.figure(figsize=(15, 6))

    for i, idx in enumerate(misclassified_idxs[:num_to_show]):
        img = generator[idx // generator.batch_size][0][idx % generator.batch_size]
        img = np.squeeze(img) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        plt.subplot(2, (num_to_show + 1) // 2, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        true_class = CLASS_NAMES[int(labels[idx])]
        pred_class = CLASS_NAMES[int(predicted_labels[idx])]
        plt.title(f"True: {true_class}\nPred: {pred_class}", fontsize=10)

    plt.tight_layout()
    plt.show()

# ====================
# Оценка модели
# ====================
def evaluate(model, test_generator, threshold=OPTIMAL_THRESHOLD):
    print("🔍 Оцениваем модель...")

    # Предсказания
    preds = model.predict(test_generator, verbose=0).flatten()
    labels = test_generator.classes
    binary_preds = (preds > threshold).astype(int)

    # Accuracy
    acc = accuracy_score(labels, binary_preds)
    print(f"\n✅ Accuracy: {acc:.4f}")

    # AUC
    auc = roc_auc_score(labels, preds)
    print(f"✅ AUC: {auc:.4f}")

    # Отчёт по precision, recall, F1
    print("\n📊 Classification Report:")
    print(classification_report(labels, binary_preds, target_names=CLASS_NAMES))

    # Матрица ошибок
    cm = confusion_matrix(labels, binary_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    show_misclassified_examples(test_generator, preds, labels, threshold)

# ====================
# Главный блок
# ====================
if __name__ == "__main__":
    print("🚀 Загрузка модели и данных...")

    test_generator = get_dataset("dataset_split/test")
    model = tf.keras.models.load_model(MODEL_PATH)

    evaluate(model, test_generator)
