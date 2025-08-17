#!/usr/bin/env python3
"""
Test all models on a single image to verify they're working
"""

import cv2
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Try importing TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    print("❌ TensorFlow not available - only traditional ML models will be tested")
    TF_AVAILABLE = False

class ModelTester:
    def __init__(self):
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available models"""
        print("\n🔄 Loading models...")
        
        # Traditional ML models
        traditional_models = [
            ('KNN', 'KNN/knn_classifier_model.pkl'),
            ('Random Forest', 'RF/rf_classifier_model.pkl'),
            ('SVC', 'SVC/svm_classifier_model.pkl')
        ]
        
        for name, path in traditional_models:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    print(f"✅ Loaded {name}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            else:
                print(f"⚠️  {name} model file not found: {path}")
        
        # Deep learning models
        if TF_AVAILABLE:
            dl_models = [
                ('CNN', 'CNN/cnn_model.h5'),
                ('VGG19', 'vgg19/arecanut_ripeness_detection.h5')
            ]
            
            for name, path in dl_models:
                if os.path.exists(path):
                    try:
                        self.models[name] = tf.keras.models.load_model(path)
                        print(f"✅ Loaded {name}")
                    except Exception as e:
                        print(f"❌ Failed to load {name}: {e}")
                else:
                    print(f"⚠️  {name} model file not found: {path}")
    
    def preprocess_for_traditional_ml(self, image):
        """Extract HSV features for traditional ML models"""
        image_resized = cv2.resize(image, (100, 100))
        hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        features = [np.mean(h), np.mean(s), np.mean(v)]
        return np.array(features).reshape(1, -1)
    
    def extract_enhanced_features(self, image):
        """Extract enhanced features for improved Random Forest"""
        features = []
        
        # Resize image
        image = cv2.resize(image, (100, 100))
        
        # 1. HSV Color Features
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        # HSV means and std
        features.extend([np.mean(h), np.mean(s), np.mean(v)])
        features.extend([np.std(h), np.std(s), np.std(v)])
        
        # 2. RGB Color Features
        b, g, r = cv2.split(image)
        features.extend([np.mean(r), np.mean(g), np.mean(b)])
        features.extend([np.std(r), np.std(g), np.std(b)])
        
        # 3. Color Ratios
        rg_ratio = np.mean(r) / (np.mean(g) + 1e-6)
        sv_ratio = np.mean(s) / (np.mean(v) + 1e-6)
        features.extend([rg_ratio, sv_ratio])
        
        # 4. Texture Features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([np.mean(grad_magnitude), np.std(grad_magnitude)])
        
        # 5. Statistical Features
        for channel in [h, s, v]:
            flat_channel = channel.flatten()
            mean_val = np.mean(flat_channel)
            std_val = np.std(flat_channel)
            
            if std_val > 0:
                skewness = np.mean(((flat_channel - mean_val) / std_val) ** 3)
                kurtosis = np.mean(((flat_channel - mean_val) / std_val) ** 4) - 3
            else:
                skewness = 0
                kurtosis = 0
                
            features.extend([skewness, kurtosis])
        
        # 6. Color Histograms
        hist_h = cv2.calcHist([h], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([s], [0], None, [8], [0, 256])
        hist_v = cv2.calcHist([v], [0], None, [8], [0, 256])
        
        hist_h = hist_h.flatten() / np.sum(hist_h)
        hist_s = hist_s.flatten() / np.sum(hist_s)
        hist_v = hist_v.flatten() / np.sum(hist_v)
        
        features.extend(hist_h.tolist())
        features.extend(hist_s.tolist())
        features.extend(hist_v.tolist())
        
        return np.array(features).reshape(1, -1)
    
    def preprocess_for_cnn(self, image):
        """Preprocess for custom CNN"""
        image_resized = cv2.resize(image, (150, 150))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb / 255.0
        return np.expand_dims(image_normalized, axis=0)
    
    def preprocess_for_vgg19(self, image):
        """Preprocess for VGG19"""
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_array = np.expand_dims(image_rgb, axis=0)
        return tf.keras.applications.vgg19.preprocess_input(image_array)
    
    def predict_single_model(self, model_name, image):
        """Make prediction with a single model"""
        if model_name not in self.models:
            return None, 0.0, "Model not loaded"
        
        try:
            if model_name in ['KNN', 'SVC']:
                # Traditional ML models with simple features
                features = self.preprocess_for_traditional_ml(image)
                model = self.models[model_name]
                prediction = model.predict(features)[0]
                
                # Get confidence (probability)
                try:
                    probabilities = model.predict_proba(features)[0]
                    confidence = max(probabilities)
                    
                    # Get class probabilities for detailed output
                    classes = model.classes_
                    prob_dict = dict(zip(classes, probabilities))
                    details = f"Probabilities: {prob_dict}"
                except:
                    confidence = 0.8  # Default confidence if predict_proba fails
                    details = "Confidence estimation not available"
                    
            elif model_name == 'Random Forest':
                # Enhanced Random Forest with more features
                features = self.extract_enhanced_features(image)
                model = self.models[model_name]
                
                # Check if we need to scale features
                scaler_path = 'RF/rf_scaler.pkl'
                if os.path.exists(scaler_path):
                    try:
                        import pickle
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                        features = scaler.transform(features)
                    except:
                        pass  # Use unscaled features if scaler fails
                
                prediction = model.predict(features)[0]
                
                # Get confidence (probability)
                try:
                    probabilities = model.predict_proba(features)[0]
                    confidence = max(probabilities)
                    
                    # Get class probabilities for detailed output
                    classes = model.classes_
                    prob_dict = dict(zip(classes, probabilities))
                    details = f"Probabilities: {prob_dict}"
                except:
                    confidence = 0.8
                    details = "Confidence estimation not available"
                
            elif model_name == 'CNN':
                # Custom CNN
                processed_image = self.preprocess_for_cnn(image)
                model = self.models[model_name]
                prediction_probs = model.predict(processed_image, verbose=0)[0]
                
                # Assuming class order: ['Ripe', 'Unripe'] - adjust if different
                class_names = ['Ripe', 'Unripe']
                prediction = class_names[np.argmax(prediction_probs)]
                confidence = max(prediction_probs)
                details = f"Raw probabilities: {prediction_probs}"
                
            elif model_name == 'VGG19':
                # VGG19 transfer learning
                processed_image = self.preprocess_for_vgg19(image)
                model = self.models[model_name]
                prediction_probs = model.predict(processed_image, verbose=0)[0]
                
                # Assuming class order: ['Unripe', 'Ripe'] - adjust if different
                class_names = ['Unripe', 'Ripe']
                prediction = class_names[np.argmax(prediction_probs)]
                confidence = max(prediction_probs)
                details = f"Raw probabilities: {prediction_probs}"
            
            return prediction, confidence, details
            
        except Exception as e:
            return None, 0.0, f"Error: {str(e)}"
    
    def test_image(self, image_path):
        """Test all models on a single image"""
        print(f"\n🖼️  Testing image: {image_path}")
        print("=" * 60)
        
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return
        
        print(f"📏 Image shape: {image.shape}")
        
        # Get ground truth from path
        if "/ripe/" in image_path.lower():
            ground_truth = "ripe"
        elif "/unripe/" in image_path.lower():
            ground_truth = "unripe"
        else:
            ground_truth = "unknown"
        print(f"🎯 Ground truth: {ground_truth.upper()}")
        print()
        
        # Test each model
        results = {}
        votes = {'ripe': 0, 'unripe': 0}
        
        for model_name in self.models.keys():
            print(f"🔍 Testing {model_name}...")
            prediction, confidence, details = self.predict_single_model(model_name, image)
            
            if prediction:
                results[model_name] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'details': details
                }
                
                # Count votes (weighted by confidence)
                vote_key = prediction.lower()
                if vote_key in votes:
                    votes[vote_key] += confidence
                
                # Check if prediction matches ground truth
                correct = "✅" if prediction.lower() == ground_truth.lower() else "❌"
                print(f"   Result: {prediction} (confidence: {confidence:.3f}) {correct}")
                print(f"   Details: {details}")
            else:
                print(f"   ❌ Failed: {details}")
            print()
        
        # Final ensemble result
        print("🗳️  ENSEMBLE VOTING RESULTS")
        print("-" * 30)
        print(f"Ripe votes: {votes['ripe']:.3f}")
        print(f"Unripe votes: {votes['unripe']:.3f}")
        
        if votes['ripe'] > votes['unripe']:
            final_prediction = 'Ripe'
        elif votes['unripe'] > votes['ripe']:
            final_prediction = 'Unripe'
        else:
            final_prediction = 'Uncertain'
        
        final_correct = "✅" if final_prediction.lower() == ground_truth.lower() else "❌"
        print(f"\n🏆 FINAL PREDICTION: {final_prediction} {final_correct}")
        
        # Summary
        correct_count = sum(1 for r in results.values() if r['prediction'].lower() == ground_truth.lower())
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 SUMMARY")
        print(f"Models tested: {total_count}")
        print(f"Correct predictions: {correct_count}")
        print(f"Individual accuracy: {accuracy:.1%}")
        print(f"Ensemble correct: {'Yes' if final_correct == '✅' else 'No'}")
        
        return results, final_prediction

def main():
    print("🥥 Areca Nut Ripeness Detection - Model Tester")
    print("=" * 50)
    
    tester = ModelTester()
    
    if not tester.models:
        print("❌ No models loaded successfully!")
        return
    
    print(f"\n✅ Successfully loaded {len(tester.models)} models:")
    for model_name in tester.models.keys():
        print(f"   - {model_name}")
    
    # Test your custom image
    test_images = [
        "single.png"  # Your areca nut image
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            tester.test_image(image_path)
            print("\n" + "="*80 + "\n")
        else:
            print(f"⚠️  Test image not found: {image_path}")
    
    print("🎉 Testing complete!")

if __name__ == "__main__":
    main()