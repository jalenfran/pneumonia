import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
import matplotlib.pyplot as plt

# ...existing code...
base_dir = "assets/"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

def load_and_preprocess_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical"
    )

def build_model():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    model = Sequential([
        data_augmentation,
        Rescaling(1./255),
        Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model

def plot_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.show()

def main():
    train_dataset = load_and_preprocess_dataset(train_dir)
    val_dataset = load_and_preprocess_dataset(val_dir)
    test_dataset = load_and_preprocess_dataset(test_dir)

    # Normalize datasets
    normalization_layer = Rescaling(1.0/255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Build and compile the model
    model = build_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    EPOCHS = 10
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

    # Save the model
    model.save("pneumonia_model.keras")

    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Plot training history
    plot_history(history)

if __name__ == "__main__":
    main()
