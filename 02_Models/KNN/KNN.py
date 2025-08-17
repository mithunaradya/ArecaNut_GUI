import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Define functions for image preprocessing
def preprocess_image(image_path, target_size=(100, 100)):
    # print(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

def extract_hsv_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)
    return [h_mean, s_mean, v_mean]

# Load and preprocess dataset
dataset_folder = "./../dataset/train"
labels = []
data = []

for label in os.listdir(dataset_folder):
    label_folder = os.path.join(dataset_folder, label)
    for image_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_file)
        image = preprocess_image(image_path)
        hsv_features = extract_hsv_features(image)
        data.append(hsv_features)
        labels.append(label)


print(data,labels)
# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize and train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Evaluate KNN classifier
accuracy = knn_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict labels for test data
predictions = knn_classifier.predict(X_test)


# Display test results
accurate_count = np.sum(predictions == y_test)
inaccurate_count = len(y_test) - accurate_count
print("Final Test Results:")
print("Total test data:", len(y_test))
print("Accurate predictions:", accurate_count)
print("Inaccurate predictions:", inaccurate_count)

with open('knn_classifier_model.pkl', 'wb') as f:
    pickle.dump(knn_classifier, f)