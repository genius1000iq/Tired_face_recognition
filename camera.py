import cv2
import numpy as np
import time
import psutil
import os
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('best_model.keras')

# Каскад Хаара
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Предобработка
def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (227, 227))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

# Захват с камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка при подключении к камере.")
    exit()

# Подготовка к замеру
proc = psutil.Process(os.getpid())
predict_count = 0
max_predictions = 100

tired_count = 0
not_tired_count = 0

total_time = 0.0
total_mem = 0.0
cpu_usages = []

print("Сбор 100 предсказаний...")

while predict_count < max_predictions:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при получении кадра.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    label = "No face"
    color = (150, 150, 150)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Увеличим рамку лица на 30% для анализа большей области
        pad = int(0.5 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_img = frame[y1:y2, x1:x2]
        input_data = preprocess_face(face_img)

        # Измерения до
        start_time = time.time()
        start_mem = proc.memory_info().rss

        prediction = model.predict(input_data, verbose=0)[0][0]

        # Измерения после
        end_time = time.time()
        end_mem = proc.memory_info().rss
        cpu = psutil.cpu_percent(interval=0.01)

        total_time += (end_time - start_time)
        total_mem += (end_mem - start_mem)
        cpu_usages.append(cpu)

        predict_count += 1

        if prediction >= 0.5:
            label = "Tired"
            color = (0, 0, 255)
            tired_count += 1
        else:
            label = "Not tired"
            color = (0, 255, 0)
            not_tired_count += 1

        # Рисуем увеличенную рамку
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Текст и счёт
    cv2.putText(frame, f"{label} ({predict_count}/{max_predictions})", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fatigue Detection", frame)

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение
cap.release()
cv2.destroyAllWindows()

# Итоговая статистика
avg_time_ms = (total_time / predict_count) * 1000
avg_mem_mb = (total_mem / predict_count) / (1024 * 1024)
avg_cpu = sum(cpu_usages) / len(cpu_usages)

print("\n=== Статистика предсказаний ===")
print(f"Всего: {predict_count}")
print(f"Tired: {tired_count} раз")
print(f"Not tired: {not_tired_count} раз")

print("\n=== Среднее использование ресурсов на одно предсказание ===")
print(f"Время: {avg_time_ms:.2f} мс")
print(f"Память: {avg_mem_mb:.2f} MB")
print(f"CPU: {avg_cpu:.2f}%")
