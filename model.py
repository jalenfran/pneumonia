import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, Input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# data directories
base_dir = "assets/"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# image and batch settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# loads data with images resized, grayscale and binary labels
def load_and_preprocess_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary"
    )

# function that punishes more for false positives
def custom_loss(alpha=10.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        false_positive_penalty = alpha * (1 - y_true) * y_pred
        return bce + false_positive_penalty
    return loss

# model architecture
def build_model():
    model = Sequential([
        Input(shape=(150, 150, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4), 
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# custom loss class for serialization
@tf.keras.utils.register_keras_serializable()
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=15.0, name="custom_loss"):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        false_positive_penalty = self.alpha * (1 - y_true) * y_pred
        return bce + false_positive_penalty

# displays loss and accuracy trends
def plot_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.show()

def main():
    # Load datasets
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
                  loss=custom_loss(alpha=8.0),
                  metrics=['accuracy'])
    # Train the model
    EPOCHS = 3
    # increases weight for negative class
    class_weights = {0: 5.0, 1: 1.0}
    early_stop = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    history = model.fit(train_dataset, 
                        validation_data=val_dataset,
                        epochs=EPOCHS,
                        class_weight=class_weights,
                        callbacks=[early_stop])
    
    # Save the model
    model.save("pneumonia_model.keras")
    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Compute confusion matrix-based metrics
    y_true = []
    y_pred = []
    threshold = 0.8
    for images, labels in test_dataset:
        preds = model.predict(images)
        # Convert predictions to binary values using the threshold
        y_pred.extend([1 if p[0] >= threshold else 0 for p in preds])
        # For binary label_mode, labels come as floats (or 0/1) in shape (batch,1)
        y_true.extend(labels.numpy().flatten().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")

    # Plots the history
    plot_history(history)

if __name__ == "__main__":
    main()
