import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

# Load the trained model
model = load_model('arecanut_ripeness_detection.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)
    return processed_img

def predict_arecanut_ripeness(image_path, threshold=0.5):
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    classes = ['Unripe',  'Ripe']
    print(predictions)
    predicted_class = classes[np.argmax(predictions)]

    # Check if the highest prediction probability is below the threshold
    if np.max(predictions) < threshold:
        return 'Uncertain'
    
    return predicted_class


def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        display_image(file_path)
        # result_label.config(text=f"Predicted Ripeness: {predict_arecanut_ripeness(file_path)}")
        result_label.config(text=f"Predicted Ripeness: {predict_arecanut_ripeness(file_path, threshold=0.7)}")


def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    image_label.img = img
    image_label.config(image=img)

# Create the main window
root = tk.Tk()
root.title("Arecanut Ripeness Prediction")

# Create and configure GUI components
open_button = tk.Button(root, text="Open Image", command=open_file_dialog)
open_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the main loop
root.mainloop()
