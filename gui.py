import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

def load_model(model_path="pneumonia_model.keras"):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, img_size=(150, 150)):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale",
                                                target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_pneumonia(model, image_path):
    processed = preprocess_image(image_path)
    prediction = model.predict(processed)
    return "Pneumonia" if np.argmax(prediction, axis=1)[0] == 1 else "Normal"

def open_image(panel, result_label, model):
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        result = predict_pneumonia(model, file_path)
        result_label.config(text=f"Prediction: {result}")

def main():
    model = load_model()
    root = tk.Tk()
    root.title("Pneumonia Classifier")
    
    panel = tk.Label(root)
    panel.pack()
    
    result_label = tk.Label(root, text="")
    result_label.pack()
    
    btn = tk.Button(root, text="Upload Image", 
                    command=lambda: open_image(panel, result_label, model))
    btn.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
