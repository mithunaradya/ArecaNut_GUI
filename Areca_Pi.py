#!/usr/bin/env python3
"""
Areca Nut Ripeness Detection - Raspberry Pi Version
Optimized GUI for Raspberry Pi with camera integration
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import os
import threading
from datetime import datetime

class ArecaPiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🥥 Areca Pi - Ripeness Detector")
        self.root.geometry("800x480")  # Pi touchscreen size
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.models = {}
        self.current_image = None
        self.camera = None
        self.camera_active = False
        
        # Create GUI
        self.create_widgets()
        
        # Load models in background
        self.load_models_async()
        
        # Initialize camera
        self.init_camera()
    
    def create_widgets(self):
        """Create the Pi-optimized GUI interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🥥 Areca Pi Detector", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Left panel - Camera and controls
        left_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 3))
        
        # Camera display area
        self.camera_frame = tk.Frame(left_frame, bg='white')
        self.camera_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.camera_label = tk.Label(self.camera_frame, text="📷\n\nInitializing Camera...", 
                                    font=('Arial', 12), fg='#7f8c8d', bg='white')
        self.camera_label.pack(expand=True)
        
        # Camera controls
        controls_frame = tk.Frame(left_frame, bg='white')
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        self.capture_btn = tk.Button(controls_frame, text="📸 Capture", font=('Arial', 11, 'bold'),
                                    bg='#3498db', fg='white', command=self.capture_image,
                                    relief='flat', padx=15, pady=8)
        self.capture_btn.pack(side='left', padx=2)
        
        self.camera_btn = tk.Button(controls_frame, text="🎥 Start Camera", font=('Arial', 11, 'bold'),
                                   bg='#27ae60', fg='white', command=self.toggle_camera,
                                   relief='flat', padx=15, pady=8)
        self.camera_btn.pack(side='left', padx=2)
        
        # Right panel - Results (smaller for Pi)
        right_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2, width=250)
        right_frame.pack(side='right', fill='y', padx=(3, 0))
        right_frame.pack_propagate(False)
        
        # Results title
        results_title = tk.Label(right_frame, text="🔍 Results", 
                                font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        results_title.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(right_frame, text="Loading models...", 
                                    font=('Arial', 9), bg='white', fg='#e67e22')
        self.status_label.pack(pady=3)
        
        # Results area (scrollable)
        results_frame = tk.Frame(right_frame, bg='white')
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
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
        self.predict_btn = tk.Button(right_frame, text="🔮 Predict", 
                                    font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
                                    command=self.predict_ripeness, relief='flat', 
                                    padx=15, pady=8, state='disabled')
        self.predict_btn.pack(pady=5)
        
        # Clear button
        clear_btn = tk.Button(right_frame, text="🗑️ Clear", 
                             font=('Arial', 9), bg='#e74c3c', fg='white',
                             command=self.clear_results, relief='flat', padx=10, pady=5)
        clear_btn.pack(pady=2)
    
    def init_camera(self):
        """Initialize the camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                # Set camera properties for Pi
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Pi
                self.camera_label.config(text="📷\n\nCamera Ready!\nClick 'Start Camera'")
            else:
                self.camera_label.config(text="❌\n\nCamera Not Found\nCheck USB connection")
        except Exception as e:
            self.camera_label.config(text=f"❌\n\nCamera Error:\n{str(e)}")
    
    def toggle_camera(self):
        """Start/stop camera preview"""
        if not self.camera_active:
            self.start_camera_preview()
        else:
            self.stop_camera_preview()
    
    def start_camera_preview(self):
        """Start camera preview"""
        if self.camera and self.camera.isOpened():
            self.camera_active = True
            self.camera_btn.config(text="⏹️ Stop Camera", bg='#e74c3c')
            self.update_camera_preview()
        else:
            messagebox.showerror("Error", "Camera not available!")
    
    def stop_camera_preview(self):
        """Stop camera preview"""
        self.camera_active = False
        self.camera_btn.config(text="🎥 Start Camera", bg='#27ae60')
        self.camera_label.config(text="📷\n\nCamera Stopped\nClick 'Start Camera'")
    
    def update_camera_preview(self):
        """Update camera preview"""
        if self.camera_active and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # Resize for display (smaller for Pi)
                frame_resized = cv2.resize(frame, (320, 240))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update label
                self.camera_label.config(image=photo, text="")
                self.camera_label.image = photo
                
                # Schedule next update
                self.root.after(100, self.update_camera_preview)  # 10 FPS for Pi
    
    def capture_image(self):
        """Capture image from camera"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_image = frame.copy()
                
                # Save captured image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_areca_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                
                # Show captured image in preview
                frame_resized = cv2.resize(frame, (320, 240))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.camera_label.config(image=photo, text="")
                self.camera_label.image = photo
                
                # Enable predict button
                if self.models:
                    self.predict_btn.config(state='normal')
                
                # Stop camera preview
                self.stop_camera_preview()
                
                messagebox.showinfo("Success", f"Image captured!\nSaved as: {filename}")
            else:
                messagebox.showerror("Error", "Failed to capture image!")
        else:
            messagebox.showerror("Error", "Camera not available!")
    
    def load_models_async(self):
        """Load models in a separate thread"""
        def load_models():
            self.load_all_models()
            self.root.after(0, self.on_models_loaded)
        
        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()
    
    def load_all_models(self):
        """Load available models (Pi optimized - no TensorFlow)"""
        # Only traditional ML models for Pi (lighter)
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
    
    def on_models_loaded(self):
        """Called when models are loaded"""
        if self.models:
            self.status_label.config(text=f"✅ {len(self.models)} models ready", fg='#27ae60')
            if self.current_image is not None:
                self.predict_btn.config(state='normal')
        else:
            self.status_label.config(text="❌ No models loaded", fg='#e74c3c')
    
    def predict_ripeness(self):
        """Predict ripeness using loaded models"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please capture an image first.")
            return
        
        if not self.models:
            messagebox.showerror("Error", "No models are loaded.")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Show loading
        loading_label = tk.Label(self.scrollable_frame, text="🔄 Analyzing...", 
                                font=('Arial', 10), bg='white', fg='#e67e22')
        loading_label.pack(pady=5)
        
        self.root.update()
        
        # Run predictions
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
                prediction, confidence = self.predict_single_model(model_name, self.current_image)
                
                if prediction:
                    results[model_name] = {
                        'prediction': prediction,
                        'confidence': confidence
                    }
                    
                    # Count votes
                    vote_key = prediction.lower()
                    if vote_key in votes:
                        votes[vote_key] += confidence
                
            except Exception as e:
                results[model_name] = {
                    'prediction': 'Error',
                    'confidence': 0.0
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
    
    def predict_single_model(self, model_name, image):
        """Make prediction with a single model"""
        if model_name not in self.models:
            return None, 0.0
        
        if model_name in ['KNN', 'SVC']:
            # Simple HSV features
            features = self.preprocess_for_traditional_ml(image)
            model = self.models[model_name]
            prediction = model.predict(features)[0]
            
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
            except:
                confidence = 0.8
                
        elif model_name == 'Random Forest':
            # Enhanced features
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
        
        return prediction, confidence
    
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
    
    def display_results(self, results):
        """Display prediction results (Pi optimized)"""
        # Clear loading message
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Ensemble result
        ensemble = results['ensemble']
        
        # Main result frame
        main_result_frame = tk.Frame(self.scrollable_frame, bg='#ecf0f1', relief='raised', bd=2)
        main_result_frame.pack(fill='x', padx=2, pady=5)
        
        # Result
        color = '#27ae60' if ensemble['prediction'] == 'Ripe' else '#e74c3c' if ensemble['prediction'] == 'Unripe' else '#f39c12'
        emoji = '🟢' if ensemble['prediction'] == 'Ripe' else '🔴' if ensemble['prediction'] == 'Unripe' else '🟡'
        
        result_text = f"{emoji} {ensemble['prediction'].upper()}"
        result_label = tk.Label(main_result_frame, text=result_text, 
                               font=('Arial', 14, 'bold'), bg='#ecf0f1', fg=color)
        result_label.pack(pady=3)
        
        # Confidence
        confidence_text = f"Confidence: {ensemble['confidence']:.1%}"
        confidence_label = tk.Label(main_result_frame, text=confidence_text, 
                                   font=('Arial', 10), bg='#ecf0f1', fg='#34495e')
        confidence_label.pack(pady=1)
        
        # Individual results (compact for Pi)
        for model_name, result in results['individual'].items():
            if result['prediction'] != 'Error':
                model_color = '#27ae60' if result['prediction'] == 'Ripe' else '#e74c3c'
                model_emoji = '🟢' if result['prediction'] == 'Ripe' else '🔴'
                text = f"{model_name}: {model_emoji} {result['confidence']:.1%}"
                
                model_label = tk.Label(self.scrollable_frame, text=text, 
                                      font=('Arial', 8), bg='white', fg=model_color)
                model_label.pack(anchor='w', padx=2, pady=1)
    
    def clear_results(self):
        """Clear all results"""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
    
    def __del__(self):
        """Cleanup camera on exit"""
        if self.camera:
            self.camera.release()

def main():
    """Main function to run the Pi GUI"""
    root = tk.Tk()
    app = ArecaPiGUI(root)
    
    # Handle window close
    def on_closing():
        if app.camera:
            app.camera.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()