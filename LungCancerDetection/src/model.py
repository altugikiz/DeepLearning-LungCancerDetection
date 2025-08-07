from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

def build_model(input_shape=(150, 150, 1), summary=True):
    """
    CNN modelini oluşturur, derler ve döndürür.
    """
    model = Sequential()
    
    model.add(Convolution2D(128, (7, 7), strides=1, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    model.add(Convolution2D(64, (5, 5), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    model.add(Convolution2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid")) # Binary classification için sigmoid

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    
    if summary:
        model.summary()
        
    return model