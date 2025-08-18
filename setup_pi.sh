#!/bin/bash
# Areca Nut Detection - Raspberry Pi Setup Script
# Run this script on your Raspberry Pi to set up everything

echo "🥥 Areca Nut Ripeness Detection - Pi Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Pi
print_step "Checking system..."
if [[ $(uname -m) == "armv7l" ]] || [[ $(uname -m) == "aarch64" ]]; then
    print_status "Running on Raspberry Pi ($(uname -m))"
else
    print_warning "This script is designed for Raspberry Pi"
fi

# Update system
print_step "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_step "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    espeak \
    espeak-data

# Install Python packages
print_step "Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements_pi.txt

# Test camera
print_step "Testing camera..."
if [ -e /dev/video0 ]; then
    print_status "Camera detected at /dev/video0"
    
    # Test camera capture
    print_status "Testing camera capture..."
    python3 -c "
import cv2
import sys
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('camera_test.jpg', frame)
        print('✅ Camera test successful! Image saved as camera_test.jpg')
    else:
        print('❌ Failed to capture image')
        sys.exit(1)
    cap.release()
else:
    print('❌ Failed to open camera')
    sys.exit(1)
"
else
    print_error "No camera detected! Please check USB connection."
fi

# Test OpenCV
print_step "Testing OpenCV installation..."
python3 -c "
import cv2
import numpy as np
print(f'✅ OpenCV version: {cv2.__version__}')
print(f'✅ NumPy version: {np.__version__}')
"

# Test scikit-learn
print_step "Testing scikit-learn installation..."
python3 -c "
import sklearn
print(f'✅ scikit-learn version: {sklearn.__version__}')
"

# Check model files
print_step "Checking model files..."
if [ -d "02_Models" ]; then
    print_status "Model directory found"
    
    # Check individual models
    models=("02_Models/KNN/knn_classifier_model.pkl" 
            "02_Models/RandomForest/rf_classifier_model.pkl" 
            "02_Models/SVC/svm_classifier_model.pkl")
    
    for model in "${models[@]}"; do
        if [ -f "$model" ]; then
            print_status "✅ Found: $model"
        else
            print_warning "❌ Missing: $model"
        fi
    done
else
    print_error "Model directory not found! Make sure you cloned the complete repository."
fi

# Create desktop shortcut
print_step "Creating desktop shortcut..."
cat > ~/Desktop/ArecaPi.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Areca Pi Detector
Comment=Areca Nut Ripeness Detection
Exec=python3 $(pwd)/Areca_Pi.py
Icon=applications-science
Terminal=false
Categories=Science;Education;
EOF

chmod +x ~/Desktop/ArecaPi.desktop

# Set up auto-start (optional)
read -p "Do you want to auto-start the application on boot? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Setting up auto-start..."
    
    # Create autostart directory if it doesn't exist
    mkdir -p ~/.config/autostart
    
    # Create autostart file
    cat > ~/.config/autostart/arecanut.desktop << EOF
[Desktop Entry]
Type=Application
Name=Areca Pi Detector
Exec=python3 $(pwd)/Areca_Pi.py
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF
    
    print_status "Auto-start configured"
fi

# Final test
print_step "Running final system test..."
python3 -c "
import cv2
import numpy as np
import sklearn
import pickle
from PIL import Image
import tkinter as tk

print('🎉 All dependencies installed successfully!')
print('📋 System Summary:')
print(f'   - OpenCV: {cv2.__version__}')
print(f'   - NumPy: {np.__version__}')
print(f'   - scikit-learn: {sklearn.__version__}')
print('   - PIL/Pillow: Available')
print('   - tkinter: Available')
print()
print('🚀 Ready to run: python3 Areca_Pi.py')
"

echo
print_status "🎉 Setup complete!"
print_status "📱 You can now run the application with: python3 Areca_Pi.py"
print_status "🖥️  Or double-click the 'Areca Pi Detector' icon on your desktop"
echo
print_warning "📝 Notes:"
print_warning "   - Make sure your camera is connected via USB"
print_warning "   - The GUI is optimized for 800x480 touchscreen"
print_warning "   - Only traditional ML models are used (no TensorFlow)"
print_warning "   - Expected accuracy: 98.7%"
echo