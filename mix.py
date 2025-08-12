import os
import random
import shutil

def shuffle_files_in_subdirs(base_dir):
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        random.shuffle(files)

        # Переименуем файлы по новому порядку
        for i, filename in enumerate(files):
            src = os.path.join(class_path, filename)
            ext = os.path.splitext(filename)[1]
            dst = os.path.join(class_path, f"{i:06d}{ext}")
            shutil.move(src, dst)

# Задаем корневые папки
for split in ['dataset_split/train', 'dataset_split/val', 'dataset_split/test']:
    shuffle_files_in_subdirs(split)
