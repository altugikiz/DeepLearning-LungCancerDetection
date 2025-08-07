import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Kendi oluşturduğumuz modülleri import edelim
from data_loader import load_and_preprocess_data
from model import build_model
from visualize import plot_training_history

# --- Yapılandırma ---
# Proje ana dizinine göre veri yolunu ayarlayın
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(BASE_DIR, "data", "chest_xray")
IMG_SIZE = 150
EPOCHS = 3
BATCH_SIZE = 32

def main():
    """Ana eğitim fonksiyonu"""
    
    # 1. Veriyi Yükle ve Ön İşle
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_and_preprocess_data(DATA_PATH, IMG_SIZE)

    # 2. Modeli Oluştur
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    model = build_model(input_shape=input_shape)

    # 3. Veri Artırma (Data Augmentation)
    datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # 4. Callback'leri Ayarla
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_accuracy", 
        patience=2, 
        verbose=1, 
        factor=0.3, 
        min_lr=0.00001
    )
    callbacks = [learning_rate_reduction]

    # 5. Modeli Eğit
    print("Model eğitimi başlıyor...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_val, y_val), # Augmentation olmadan doğrulama
        callbacks=callbacks
    )

    # 6. Modeli Değerlendir
    print("\nModel test verisi üzerinde değerlendiriliyor...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Verisi Kaybı (Loss): {loss:.4f}")
    print(f"Test Verisi Başarımı (Accuracy): {accuracy*100:.2f}%")

    # 7. Sonuçları Görselleştir ve Kaydet
    plot_training_history(history, save_path=os.path.join(BASE_DIR, "training_history.png"))

    # 8. Eğitilmiş Modeli Kaydet
    model.save(os.path.join(BASE_DIR, "pneumonia_classifier_model.h5"))
    print(f"Eğitilmiş model 'pneumonia_classifier_model.h5' olarak kaydedildi.")


if __name__ == "__main__":
    main()