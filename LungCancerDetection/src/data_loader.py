import os
import cv2
import numpy as np
from tqdm import tqdm

LABELS = ["PNEUMONIA", "NORMAL"]

def _load_images_from_folder(data_dir, img_size):
    """Belirtilen klasörden resimleri ve etiketleri yükler."""
    data = []
    for label in LABELS:
        path = os.path.join(data_dir, label)
        class_num = LABELS.index(label)
        for img in tqdm(os.listdir(path), desc=f"Loading {os.path.basename(data_dir)} images for {label}"):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    print(f"Uyarı: {os.path.join(path, img)} dosyası okunamadı.")
                    continue
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Hata: {os.path.join(path, img)} işlenirken hata oluştu - {e}")
    return np.array(data, dtype=object)

def load_and_preprocess_data(base_path, img_size=150):
    """
    Train, test ve val veri setlerini yükler, ayırır ve ön işler.
    
    Returns:
        Tuple: ((x_train, y_train), (x_test, y_test), (x_val, y_val))
    """
    print("Veri yükleme ve ön işleme süreci başladı...")
    
    # Veri setlerini yükle
    train_data = _load_images_from_folder(os.path.join(base_path, "train"), img_size)
    test_data = _load_images_from_folder(os.path.join(base_path, "test"), img_size)
    val_data = _load_images_from_folder(os.path.join(base_path, "val"), img_size)

    # Verileri resimler (x) ve etiketler (y) olarak ayır
    x_train = np.array([i[0] for i in train_data])
    y_train = np.array([i[1] for i in train_data])

    x_test = np.array([i[0] for i in test_data])
    y_test = np.array([i[1] for i in test_data])

    x_val = np.array([i[0] for i in val_data])
    y_val = np.array([i[1] for i in val_data])

    # Normalizasyon (Piksel değerlerini 0-1 arasına ölçekle)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_val = x_val / 255.0

    # CNN için yeniden şekillendirme (Kanal boyutu ekle)
    x_train = x_train.reshape(-1, img_size, img_size, 1)
    x_test = x_test.reshape(-1, img_size, img_size, 1) # <-- HATA BURADA DÜZELTİLDİ
    x_val = x_val.reshape(-1, img_size, img_size, 1)   # <-- HATA BURADA DÜZELTİLDİ
    
    print("Veri yükleme ve ön işleme tamamlandı.")
    
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)