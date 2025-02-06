import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from model import CustomLoss
from tkinter import ttk

# loads the model in
def load_model(model_path="pneumonia_model.keras"):
    return tf.keras.models.load_model(model_path, custom_objects={'loss': CustomLoss})

# prepares the image for prediction
def preprocess_image(image_path, img_size=(150, 150)):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale",
                                                target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# function to predict pneumonia or not
def predict_pneumonia(model, image_path, threshold = 0.8):
    processed = preprocess_image(image_path)
    prediction = model.predict(processed)
    return "Pneumonia" if prediction[0][0] >= threshold else "Normal"

# function handles opening images and displaying prediction
def open_image(panel, result_label, model):
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        result = predict_pneumonia(model, file_path, 0.8)
        result_label.config(text=f"Prediction: {result}")

def main():
    model = load_model()

    # creates GUI window
    root = tk.Tk()
    root.title("Pneumonia Classifier")
    root.geometry("800x600") 
    root.configure(bg="#f0f0f0")
    style = ttk.Style(root)
    style.theme_use('clam')
    
    # Create a main frame with padding
    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.pack(fill='both', expand=True)
    
    # Image panel
    panel = ttk.Label(main_frame)
    panel.pack(pady=10)
    
    result_label = ttk.Label(main_frame, text="", font=("Helvetica", 12))
    result_label.pack(pady=5)
    
    # upload button
    btn = ttk.Button(main_frame, text="Upload Image",
                     command=lambda: open_image(panel, result_label, model))
    btn.pack(pady=10)

    # displays author tag
    author_label = ttk.Label(main_frame, text="Written by Jalen Francis", font=("Helvetica", 20), foreground="black")
    author_label.pack(side="bottom", pady=(20, 0))
    
    root.mainloop()

if __name__ == "__main__":
    main()
