
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import numpy as np
# import tensorflow as tf

# # Load the trained model


# # Define image dimensions
# image_height, image_width = 150, 150

# # Define ripeness levels
# ripeness_levels = ['Ripe',  'Unripe']

# # Function to preprocess and predict image
# def predict_image():
#     file_path = filedialog.askopenfilename()
#     image = Image.open(file_path)
#     image = image.resize((image_height, image_width))
#     image = np.expand_dims(image, axis=0)
#     image = image / 255.0  # Normalize image
    
#     prediction = model.predict(image)
#     predicted_class = ripeness_levels[np.argmax(prediction)]
#     result_label.config(text=f'Predicted Ripeness: {predicted_class}')

# # Create Tkinter window
# window = tk.Tk()
# window.title('Arecanut Ripeness Detector')

# # Create browse button
# browse_button = tk.Button(window, text='Browse', command=predict_image)
# browse_button.pack(pady=10)

# # Create label for result
# result_label = tk.Label(window, text='')
# result_label.pack(pady=10)

# window.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Define image dimensions
image_height, image_width = 150, 150

# Define ripeness levels
ripeness_levels = ['Ripe', 'Unripe']

# Function to preprocess and predict image
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = Image.open(file_path)
            image = image.resize((image_height, image_width))
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo  # Keep a reference
            image = np.array(image)
            image = image / 255.0  # Normalize image
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            predicted_class = ripeness_levels[np.argmax(prediction)]
            result_label.config(text=f'Predicted Ripeness: {predicted_class}')
        except Exception as e:
            messagebox.showerror("Error", f"Error while processing image: {str(e)}")

# Create Tkinter window
window = tk.Tk()
window.title('Arecanut Ripeness Detector')

# Create browse button
browse_button = tk.Button(window, text='Browse', command=predict_image)
browse_button.pack(pady=10)

# Create label for image
image_label = tk.Label(window)
image_label.pack(pady=10)

# Create label for result
result_label = tk.Label(window, text='')
result_label.pack(pady=10)

window.mainloop()
