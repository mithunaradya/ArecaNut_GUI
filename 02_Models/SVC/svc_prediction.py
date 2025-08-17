# import cv2
# import numpy as np
# import pickle

# # Define functions for image preprocessing
# def preprocess_image(image_path, target_size=(100, 100)):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, target_size)
#     return image

# def extract_hsv_features(image):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv_image)
#     h_mean = np.mean(h)
#     s_mean = np.mean(s)
#     v_mean = np.mean(v)
#     return [h_mean, s_mean, v_mean]

# # Load SVM classifier from file
# with open('svm_classifier_model.pkl', 'rb') as f:
#     svm_classifier = pickle.load(f)

# # Load Random Forest classifier from file
# with open('rf_classifier_model.pkl', 'rb') as f:
#     rf_classifier = pickle.load(f)

# # Function to predict using SVM classifier
# def predict_svm(image_path):
#     image = preprocess_image(image_path)
#     features = extract_hsv_features(image)
#     features = np.array(features).reshape(1, -1)
#     predicted_label = svm_classifier.predict(features)
#     return predicted_label[0]

# # Function to predict using Random Forest classifier
# def predict_rf(image_path):
#     image = preprocess_image(image_path)
#     features = extract_hsv_features(image)
#     features = np.array(features).reshape(1, -1)
#     predicted_label = rf_classifier.predict(features)
#     return predicted_label[0]

# # Example usage
# if __name__ == "__main__":
#     image_path = "path_to_your_image.jpg"
    
#     svm_prediction = predict_svm(image_path)
#     print("SVM Prediction:", svm_prediction)

#     rf_prediction = predict_rf(image_path)
#     print("Random Forest Prediction:", rf_prediction)


import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pickle
from PIL import Image, ImageTk

class ImageClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier")
        self.geometry("400x300")

        self.svm_classifier = self.load_classifier('svm_classifier_model.pkl')
        

        self.create_widgets()

    def load_classifier(self, model_file):
        try:
            with open(model_file, 'rb') as f:
                classifier = pickle.load(f)
            return classifier
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load classifier: {e}")
            self.destroy()

    def create_widgets(self):
        self.select_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.label_result_svm = tk.Label(self, text="")
        self.label_result_svm.pack(pady=5)

       

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

    def select_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            self.display_image(image_path)

            svm_prediction = self.predict_svm(image_path)
            self.label_result_svm.config(text=f"SVM Prediction: {svm_prediction}")

           

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.image_label.img = img
        self.image_label.config(image=img)

    def preprocess_image(self, image_path, target_size=(100, 100)):
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)
        return image

    def extract_hsv_features(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)
        return [h_mean, s_mean, v_mean]

    def predict_svm(self, image_path):
        image = self.preprocess_image(image_path)
        features = self.extract_hsv_features(image)
        features = np.array(features).reshape(1, -1)
        predicted_label = self.svm_classifier.predict(features)
        return predicted_label[0]

    

if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
