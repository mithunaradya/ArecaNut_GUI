#!/usr/bin/env python3
"""
Areca Nut Ripeness Detection GUI
A user-friendly interface for predicting areca nut ripeness
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import os
import threading
from datetime import datetime

# Try importing TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class ArecaRipenessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🥥 Areca Nut Ripeness Detector")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.models = {}
        self.current_image = None
        self.current_image_path = None
        
        # Create GUI
        self.create_widgets()
        
        # Load models in background
        self.load_models_async()
    
    def create_widgets(self):
        """Create the GUI interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🥥 Areca Nut Ripeness Detector", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Image display
        left_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Image display area
        self.image_frame = tk.Frame(left_frame, bg='white')
        self.image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(self.image_frame, text="📷\n\nClick 'Upload Image' to select\nan areca nut image", 
                                   font=('Arial', 14), fg='#7f8c8d', bg='white')
        self.image_label.pack(expand=True)
        
        # Upload button
        upload_btn = tk.Button(left_frame, text="📁 Upload Image", font=('Arial', 12, 'bold'),
                              bg='#3498db', fg='white', command=self.upload_image,
                              relief='flat', padx=20, pady=10)
        upload_btn.pack(pady=10)
        
        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2, width=300)
        right_frame.pack(side='right', fill='y', padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Results title
        results_title = tk.Label(right_frame, text="🔍 Prediction Results", 
                                font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        results_title.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(right_frame, text="Loading models...", 
                                    font=('Arial', 10), bg='white', fg='#e67e22')
        self.status_label.pack(pady=5)
        
        # Results area
        results_frame = tk.Frame(right_frame, bg='white')
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollable results
        canvas = tk.Canvas(results_frame, bg='white')
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='white')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Predict button
        self.predict_btn = tk.Button(right_frame, text="🔮 Predict Ripeness", 
                                    font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                    command=self.predict_ripeness, relief='flat', 
                                    padx=20, pady=10, state='disabled')
        self.predict_btn.pack(pady=10)
        
        # Clear results button
        clear_btn = tk.Button(right_frame, text="🗑️ Clear Results", 
                             font=('Arial', 10), bg='#e74c3c', fg='white',
                             command=self.clear_results, relief='flat', padx=15, pady=5)
        clear_btn.pack(pady=5)
    
    def load_models_async(self):
        """Load models in a separate thread"""
        def load_models():
            self.load_all_models()
            self.root.after(0, self.on_models_loaded)
        
        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()
    
    def load_all_models(self):
        """Load all available models"""
        # Traditional ML models
        traditional_models = [
            ('KNN', '02_Models/KNN/knn_classifier_model.pkl'),
            ('Random Forest', '02_Models/RandomForest/rf_classifier_model.pkl'),
            ('SVC', '02_Models/SVC/svm_classifier_model.pkl')
        ]
        
        for name, path in traditional_models:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
        
        # Deep learning models
        if TF_AVAILABLE:
            dl_models = [
                ('CNN', '02_Models/CNN/cnn_model.h5')
            ]
            
            for name, path in dl_models:
                if os.path.exists(path):
                    try:
                        self.models[name] = tf.keras.models.load_model(path)
                    except Exception as e:
                        print(f"Failed to load {name}: {e}")
    
    def on_models_loaded(self):
        """Called when models are loaded"""
        if self.models:
            self.status_label.config(text=f"✅ {len(self.models)} models loaded", fg='#27ae60')
            if self.current_image is not None:
                self.predict_btn.config(state='normal')
        else:
            self.status_label.config(text="❌ No models loaded", fg='#e74c3c')
    
    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Areca Nut Image",
            filetypes=file_types
        )
        
        if file_path:
            self.load_and_display_image(file_path)
    
    def load_and_display_image(self, file_path):
        """Load and display the selected image"""
        try:
            # Load image with OpenCV
            self.current_image = cv2.imread(file_path)
            self.current_image_path = file_path
            
            if self.current_image is None:
                messagebox.showerror("Error", "Failed to load image. Please select a valid image file.")
                return
            
            # Display image in GUI
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Resize for display
            display_size = (300, 300)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Enable predict button if models are loaded
            if self.models:
                self.predict_btn.config(state='normal')
            
            # Clear previous results
            self.clear_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def predict_ripeness(self):
        """Predict ripeness using all models"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        if not self.models:
            messagebox.showerror("Error", "No models are loaded.")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Show loading
        loading_label = tk.Label(self.scrollable_frame, text="🔄 Analyzing image...", 
                                font=('Arial', 12), bg='white', fg='#e67e22')
        loading_label.pack(pady=10)
        
        self.root.update()
        
        # Run predictions in background
        def run_predictions():
            results = self.get_all_predictions()
            self.root.after(0, lambda: self.display_results(results))
        
        thread = threading.Thread(target=run_predictions, daemon=True)
        thread.start()
    
    def get_all_predictions(self):
        """Get predictions from all models"""
        results = {}
        votes = {'ripe': 0, 'unripe': 0}
        
        for model_name in self.models.keys():
            try:
                prediction, confidence, details = self.predict_single_model(model_name, self.current_image)
                
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
                
            except Exception as e:
                results[model_name] = {
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'details': str(e)
                }
        
        # Calculate ensemble result
        if votes['ripe'] > votes['unripe']:
            final_prediction = 'Ripe'
            final_confidence = votes['ripe'] / (votes['ripe'] + votes['unripe'])
        elif votes['unripe'] > votes['ripe']:
            final_prediction = 'Unripe'
            final_confidence = votes['unripe'] / (votes['ripe'] + votes['unripe'])
        else:
            final_prediction = 'Uncertain'
            final_confidence = 0.5
        
        return {
            'individual': results,
            'ensemble': {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'votes': votes
            }
        }
    
    def display_results(self, results):
        """Display prediction results"""
        # Clear loading message
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Ensemble result (main result)
        ensemble = results['ensemble']
        
        # Main result frame
        main_result_frame = tk.Frame(self.scrollable_frame, bg='#ecf0f1', relief='raised', bd=2)
        main_result_frame.pack(fill='x', padx=5, pady=10)
        
        # Ensemble title
        ensemble_title = tk.Label(main_result_frame, text="🏆 FINAL PREDICTION", 
                                 font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        ensemble_title.pack(pady=5)
        
        # Result
        color = '#27ae60' if ensemble['prediction'] == 'Ripe' else '#e74c3c' if ensemble['prediction'] == 'Unripe' else '#f39c12'
        emoji = '🟢' if ensemble['prediction'] == 'Ripe' else '🔴' if ensemble['prediction'] == 'Unripe' else '🟡'
        
        result_text = f"{emoji} {ensemble['prediction'].upper()}"
        result_label = tk.Label(main_result_frame, text=result_text, 
                               font=('Arial', 16, 'bold'), bg='#ecf0f1', fg=color)
        result_label.pack(pady=5)
        
        # Confidence
        confidence_text = f"Confidence: {ensemble['confidence']:.1%}"
        confidence_label = tk.Label(main_result_frame, text=confidence_text, 
                                   font=('Arial', 11), bg='#ecf0f1', fg='#34495e')
        confidence_label.pack(pady=2)
        
        # Voting details
        votes = ensemble['votes']
        votes_text = f"Votes - Ripe: {votes['ripe']:.2f} | Unripe: {votes['unripe']:.2f}"
        votes_label = tk.Label(main_result_frame, text=votes_text, 
                              font=('Arial', 9), bg='#ecf0f1', fg='#7f8c8d')
        votes_label.pack(pady=2)
        
        # Individual model results
        individual_title = tk.Label(self.scrollable_frame, text="📊 Individual Model Results", 
                                   font=('Arial', 11, 'bold'), bg='white', fg='#2c3e50')
        individual_title.pack(pady=(15, 5))
        
        for model_name, result in results['individual'].items():
            model_frame = tk.Frame(self.scrollable_frame, bg='white', relief='groove', bd=1)
            model_frame.pack(fill='x', padx=5, pady=2)
            
            # Model name and result
            if result['prediction'] != 'Error':
                model_color = '#27ae60' if result['prediction'] == 'Ripe' else '#e74c3c'
                model_emoji = '🟢' if result['prediction'] == 'Ripe' else '🔴'
                header_text = f"{model_name}: {model_emoji} {result['prediction']} ({result['confidence']:.1%})"
            else:
                model_color = '#e74c3c'
                header_text = f"{model_name}: ❌ Error"
            
            model_label = tk.Label(model_frame, text=header_text, 
                                  font=('Arial', 10, 'bold'), bg='white', fg=model_color)
            model_label.pack(anchor='w', padx=5, pady=2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_label = tk.Label(self.scrollable_frame, text=f"Analysis completed: {timestamp}", 
                             font=('Arial', 8), bg='white', fg='#95a5a6')
        time_label.pack(pady=10)
    
    def predict_single_model(self, model_name, image):
        """Make prediction with a single model"""
        if model_name not in self.models:
            return None, 0.0, "Model not loaded"
        
        if model_name in ['KNN', 'SVC']:
            # Traditional ML models with simple features
            features = self.preprocess_for_traditional_ml(image)
            model = self.models[model_name]
            prediction = model.predict(features)[0]
            
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
            except:
                confidence = 0.8
                
        elif model_name == 'Random Forest':
            # Enhanced Random Forest
            features = self.extract_enhanced_features(image)
            model = self.models[model_name]
            
            # Try to load scaler
            scaler_path = '02_Models/RandomForest/rf_scaler.pkl'
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    features = scaler.transform(features)
                except:
                    pass
            
            prediction = model.predict(features)[0]
            
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
            except:
                confidence = 0.8
                
        elif model_name == 'CNN':
            # Custom CNN
            processed_image = self.preprocess_for_cnn(image)
            model = self.models[model_name]
            prediction_probs = model.predict(processed_image, verbose=0)[0]
            
            class_names = ['Ripe', 'Unripe']
            prediction = class_names[np.argmax(prediction_probs)]
            confidence = max(prediction_probs)
            
        elif model_name == 'VGG19':
            # VGG19 transfer learning
            processed_image = self.preprocess_for_vgg19(image)
            model = self.models[model_name]
            prediction_probs = model.predict(processed_image, verbose=0)[0]
            
            class_names = ['Unripe', 'Ripe']
            prediction = class_names[np.argmax(prediction_probs)]
            confidence = max(prediction_probs)
        
        return prediction, confidence, "Success"
    
    def preprocess_for_traditional_ml(self, image):
        """Extract HSV features for traditional ML models"""
        image_resized = cv2.resize(image, (100, 100))
        hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        features = [np.mean(h), np.mean(s), np.mean(v)]
        return np.array(features).reshape(1, -1)
    
    def extract_enhanced_features(self, image):
        """Extract enhanced features for Random Forest"""
        features = []
        image = cv2.resize(image, (100, 100))
        
        # HSV Color Features
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        features.extend([np.mean(h), np.mean(s), np.mean(v)])
        features.extend([np.std(h), np.std(s), np.std(v)])
        
        # RGB Color Features
        b, g, r = cv2.split(image)
        features.extend([np.mean(r), np.mean(g), np.mean(b)])
        features.extend([np.std(r), np.std(g), np.std(b)])
        
        # Color Ratios
        rg_ratio = np.mean(r) / (np.mean(g) + 1e-6)
        sv_ratio = np.mean(s) / (np.mean(v) + 1e-6)
        features.extend([rg_ratio, sv_ratio])
        
        # Texture Features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([np.mean(grad_magnitude), np.std(grad_magnitude)])
        
        # Statistical Features
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
        
        # Color Histograms
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
    
    def clear_results(self):
        """Clear all results"""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = ArecaRipenessGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()