# import cv2
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
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

# # Load the trained KNN model
# # knn_classifier = KNeighborsClassifier(n_neighbors=3)
# # knn_classifier.load_model("knn_classifier_model.pkl")

# with open('knn_classifier_model.pkl', 'rb') as f:
#     knn_classifier = pickle.load(f)

# # Preprocess and extract features from the new image
# new_image_path = "./dataset/test/unripe/20231129_131529.jpg"
# new_image = preprocess_image(new_image_path)
# new_features = extract_hsv_features(new_image)

# # Reshape features into a 2D array for prediction
# new_features = np.array(new_features).reshape(1, -1)

# # Predict the label of the new image using the trained model
# predicted_label = knn_classifier.predict(new_features)

# print("Predicted label:", predicted_label)




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

        self.knn_classifier = self.load_classifier()

        self.create_widgets()

    def load_classifier(self):
        try:
            with open('knn_classifier_model.pkl', 'rb') as f:
                knn_classifier = pickle.load(f)
            return knn_classifier
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load classifier: {e}")
            self.destroy()

    def create_widgets(self):
        self.select_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.label_result = tk.Label(self, text="")
        self.label_result.pack(pady=5)

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

    def select_image(self):
        self.selected_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.selected_image_path:
            self.display_image_and_predict(self.selected_image_path)

    def display_image_and_predict(self, image_path):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (200, 200))
            photo = self.convert_to_photoimage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Predict the label of the selected image
            predicted_label = self.predict_image(image)
            self.label_result.config(text=f"Predicted: {predicted_label}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def convert_to_photoimage(self, image):
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        return photo

    def predict_image(self, image):
        image = cv2.resize(image, (100, 100))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        features = np.array([[h_mean, s_mean, v_mean]])
        predicted_label = self.knn_classifier.predict(features)
        return predicted_label[0]

if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
