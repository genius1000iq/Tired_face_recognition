import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# ====================
# ХАРДКОД-НАСТРОЙКИ
# ====================
DATASET_PATHS = {
    "train": "dataset_split/train",
    "valid": "dataset_split/val",  # добавлена валидационная выборка
    "test": "dataset_split/test"
}
HASH_ALGO = "md5"
OUTPUT_FILE = "cross_leakage_report.txt"

def get_file_hash(filepath: str) -> str:
    """Вычисляет хэш-сумму файла"""
    hash_func = getattr(hashlib, HASH_ALGO)()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def scan_all_datasets() -> dict:
    """Сканирует все датасеты и возвращает словарь {хэш: {dataset_type: [пути]}}"""
    all_hashes = defaultdict(lambda: defaultdict(list))
    
    for dataset_type, dataset_path in DATASET_PATHS.items():
        if not os.path.exists(dataset_path):
            print(f"⚠️ Директория {dataset_path} не существует! Пропускаю...")
            continue
            
        files_to_process = []
        for root, _, files in os.walk(dataset_path):
            files_to_process.extend(os.path.join(root, f) for f in files)
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_file_hash, fp): fp for fp in files_to_process}
            for future in tqdm(futures, desc=f"Сканирую {dataset_type}", unit="file"):
                filepath = futures[future]
                file_hash = future.result()
                all_hashes[file_hash][dataset_type].append(filepath)
    
    return all_hashes

def analyze_leakages(all_hashes: dict) -> dict:
    """Анализирует пересечения между датасетами"""
    leakages = defaultdict(list)
    
    for file_hash, datasets in all_hashes.items():
        if len(datasets) > 1:  # Файл есть в нескольких датасетах
            leakage = {
                "hash": file_hash,
                "locations": datasets
            }
            # Ключ для группировки (например "train-test" или "train-valid-test")
            datasets_key = "-".join(sorted(datasets.keys()))
            leakages[datasets_key].append(leakage)
    
    return leakages

def save_report(leakages: dict) -> None:
    """Генерирует подробный отчёт"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Отчёт о пересечениях в датасетах ({timestamp})\n\n")
        
        for dataset_key in DATASET_PATHS:
            f.write(f"{dataset_key}: {DATASET_PATHS[dataset_key]}\n")
        f.write(f"\nАлгоритм хэширования: {HASH_ALGO}\n")
        
        total_leakages = sum(len(v) for v in leakages.values())
        f.write(f"\nВсего пересечений: {total_leakages}\n")
        
        for datasets_key, items in leakages.items():
            f.write(f"\n=== Пересечения {datasets_key} ({len(items)} шт.) ===\n")
            for i, item in enumerate(items, 1):
                f.write(f"\nДубликат #{i} (хэш: {item['hash']}):\n")
                for ds_type, paths in item['locations'].items():
                    f.write(f"В {ds_type}:\n")
                    for path in paths:
                        f.write(f"  - {path}\n")
    
    print(f"\n✅ Отчёт сохранён в {OUTPUT_FILE}")

def main():
    print("🔍 Запуск проверки пересечений между датасетами")
    print("▸ train:", DATASET_PATHS["train"])
    print("▸ valid:", DATASET_PATHS["valid"])
    print("▸ test:", DATASET_PATHS["test"])
    
    all_hashes = scan_all_datasets()
    leakages = analyze_leakages(all_hashes)
    
    if not leakages:
        print("\n✅ Пересечений между датасетами не обнаружено!")
        return
    
    print("\n⚠️ Обнаружены пересечения:")
    for datasets_key, items in leakages.items():
        print(f"▸ {datasets_key}: {len(items)} файлов")
    
    save_report(leakages)
    
    # Дополнительная статистика
    print("\n📊 Статистика:")
    unique_files = sum(len(paths) for paths in all_hashes.values())
    print(f"Всего уникальных файлов: {len(all_hashes)}")
    print(f"Всего файлов с учётом дубликатов: {unique_files}")

if __name__ == "__main__":
    main()