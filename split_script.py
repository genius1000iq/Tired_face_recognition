import os
import shutil
import random

source_dir = r"D:\studying\SUAI\4course\diplom\Drowsy_detection_3_calc_metrics\dataset"
target_dir = r"D:\studying\SUAI\4course\diplom\Drowsy_detection_3_calc_metrics\right_split_dataset"
splits = ['train', 'val', 'test']
split_ratio = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test

random.seed(42)

# Сканируем все подпапки tired и not_tired
for class_name in ['tired', 'not_tired']:
    class_path = os.path.join(source_dir, class_name)
    files = os.listdir(class_path)

    # Группируем файлы по первой букве (предполагаем, что это идентификатор человека)
    person_dict = {}
    for fname in files:
        person_id = fname[0]  # Можно заменить на fname[:1] или другой способ, если ID длиннее
        person_dict.setdefault(person_id, []).append(fname)

    person_ids = list(person_dict.keys())
    random.shuffle(person_ids)

    # Разделяем людей по выборкам
    n_total = len(person_ids)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    train_ids = person_ids[:n_train]
    val_ids = person_ids[n_train:n_train + n_val]
    test_ids = person_ids[n_train + n_val:]

    split_map = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    # Распределяем файлы
    for split in splits:
        split_dir = os.path.join(target_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)

        for pid in split_map[split]:
            for fname in person_dict[pid]:
                src_path = os.path.join(class_path, fname)
                dst_path = os.path.join(split_dir, fname)
                shutil.copyfile(src_path, dst_path)
