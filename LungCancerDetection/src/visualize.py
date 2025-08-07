import matplotlib.pyplot as plt

def plot_training_history(history, save_path="training_history.png"):
    """
    Modelin eğitim ve doğrulama geçmişini (accuracy/loss) çizer ve kaydeder.
    """
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy Grafiği
    ax[0].plot(epochs, history.history['accuracy'], "go-", label="Eğitim Başarımı (Accuracy)")
    ax[0].plot(epochs, history.history['val_accuracy'], "ro-", label="Doğrulama Başarımı (Validation Accuracy)")
    ax[0].set_title("Eğitim ve Doğrulama Başarımı")
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    # Loss Grafiği
    ax[1].plot(epochs, history.history['loss'], "go-", label="Eğitim Kaybı (Loss)")
    ax[1].plot(epochs, history.history['val_loss'], "ro-", label="Doğrulama Kaybı (Validation Loss)")
    ax[1].set_title("Eğitim ve Doğrulama Kaybı")
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Eğitim grafiği '{save_path}' olarak kaydedildi.")
    plt.close()