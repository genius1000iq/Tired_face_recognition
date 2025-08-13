# Tired_face_recognition
This project performs a binary classification of faces into the classes 'tired' and 'not tired', using a CNN.

Файл split_script.py – скрипт для разделения набора данных на 3 выборки: тренировочную, валидационную и тестовую.
Файл mix.py – скрипт для перемешивания файлов внутри директорий с выборками.
Файл train_easy.py – файл, в котором описывается архитектура нейросети и процесс обучения.
Eval.py – файл с программой для оценки качества нейронной сети.
check_hash.py – Файл с программой для поиска дублирующих файлов в выборках
camera.py - детекция в реальном времени


При обучении я использовал датасет DDD (Drawsy detection dataset) - https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset
Для обучения модели запустить файл train_easy.py
Обученная модель сохраняется в корневую папку проекта под названием best_model.
