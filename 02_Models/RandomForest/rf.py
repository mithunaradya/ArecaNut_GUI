import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Define functions for image preprocessing
def preprocess_image(image_path, target_size=(100, 100)):
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

# Load the dataset and extract features
# Assuming you have your dataset structured appropriately

# X: Feature matrix
# y: Labels
X = []
y = []

# Iterate over your dataset to extract features and labels
# Example structure: dataset/unripe/, dataset/ripe/
for class_label in ['unripe', 'ripe']:
    images_path = f'./../dataset/train/{class_label}/'
    for image_filename in os.listdir(images_path):
        image_path = os.path.join(images_path, image_filename)
        image = preprocess_image(image_path)
        features = extract_hsv_features(image)
        X.append(features)
        y.append(class_label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define and train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Save Random Forest classifier to a file
with open('rf_classifier_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)
