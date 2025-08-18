# 🥥 Areca Nut Detection - Raspberry Pi Guide

Complete guide for running the Areca Nut Ripeness Detection system on Raspberry Pi with camera integration.

## 🎯 Quick Start

### 1. Clone Repository on Pi
```bash
git clone https://github.com/mithunaradya/ArecaNut_GUI.git
cd ArecaNut_GUI
```

### 2. Run Setup Script
```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

### 3. Launch Application
```bash
python3 Areca_Pi.py
```

## 📱 Pi-Optimized Features

### 🎥 **Camera Integration**
- **Live camera preview** with USB webcam support
- **One-click capture** for instant image analysis
- **Automatic image saving** with timestamps
- **Optimized for Pi performance** (15 FPS, 640x480)

### 🖥️ **Touchscreen Friendly**
- **800x480 resolution** optimized for Pi touchscreen
- **Large buttons** for easy touch interaction
- **Compact layout** maximizing screen space
- **Responsive interface** with visual feedback

### ⚡ **Performance Optimized**
- **Traditional ML only** (no TensorFlow - too heavy for Pi)
- **3 models**: KNN, Random Forest, SVC
- **98.7% accuracy** maintained without deep learning
- **Low memory footprint** suitable for Pi 4
- **Fast prediction** (< 2 seconds per image)

## 🛠️ Hardware Setup

### Required Hardware
- **Raspberry Pi 4** (2GB+ RAM recommended)
- **USB Webcam** (Logitech 1080p or similar)
- **MicroSD Card** (16GB+ Class 10)
- **Power Supply** (Official Pi 4 adapter)

### Optional Hardware
- **7" Touchscreen Display** for portable operation
- **Case with camera mount** for field use
- **External speakers** for audio feedback

## 📋 Software Requirements

### System Packages
```bash
sudo apt install -y python3-pip python3-opencv libatlas-base-dev
```

### Python Packages
```bash
pip3 install opencv-python scikit-learn numpy Pillow
```

## 🚀 Installation Methods

### Method 1: Automated Setup (Recommended)
```bash
git clone https://github.com/mithunaradya/ArecaNut_GUI.git
cd ArecaNut_GUI
chmod +x setup_pi.sh
./setup_pi.sh
```

### Method 2: Manual Installation
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install -y python3-pip python3-opencv libatlas-base-dev

# 3. Install Python packages
pip3 install -r requirements_pi.txt

# 4. Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# 5. Run application
python3 Areca_Pi.py
```

## 📱 Using the Application

### Step 1: Launch
```bash
python3 Areca_Pi.py
```

### Step 2: Start Camera
- Click **"🎥 Start Camera"** to begin live preview
- Position areca nut in camera view
- Ensure good lighting for best results

### Step 3: Capture Image
- Click **"📸 Capture"** when ready
- Image will be automatically saved with timestamp
- Camera preview will stop after capture

### Step 4: Predict Ripeness
- Click **"🔮 Predict"** to analyze the captured image
- Wait for analysis (usually 1-2 seconds)
- View results in the right panel

### Step 5: Interpret Results
- **🟢 Ripe**: Ready for harvest/processing
- **🔴 Unripe**: Needs more time to mature
- **Confidence %**: Model certainty (higher is better)

## 🎨 Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│              🥥 Areca Pi Detector                       │
├─────────────────────────────┬───────────────────────────┤
│                             │     🔍 Results            │
│     📷 Camera Preview       │                           │
│        (320x240)            │   Status: ✅ 3 models    │
│                             │                           │
│                             │   ┌─────────────────────┐ │
│                             │   │                     │ │
│                             │   │   Results Panel     │ │
│                             │   │   (Scrollable)      │ │
│                             │   │                     │ │
│                             │   └─────────────────────┘ │
│                             │                           │
│ 📸 Capture  🎥 Start Camera │   🔮 Predict  🗑️ Clear   │
└─────────────────────────────┴───────────────────────────┘
```

## 🔧 Configuration

### Camera Settings
Edit `Areca_Pi.py` to adjust camera parameters:
```python
# Camera resolution (lower = faster)
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame rate (lower = less CPU usage)
self.camera.set(cv2.CAP_PROP_FPS, 15)
```

### Display Settings
For different screen sizes, modify:
```python
# Window size
self.root.geometry("800x480")  # Change to your screen size

# Preview size
frame_resized = cv2.resize(frame, (320, 240))  # Adjust preview size
```

## 🐛 Troubleshooting

### Camera Issues
**Problem**: "Camera not found"
```bash
# Check camera connection
lsusb | grep -i camera

# Test camera manually
fswebcam test.jpg

# Check permissions
sudo usermod -a -G video $USER
```

**Problem**: "Permission denied"
```bash
# Add user to video group
sudo usermod -a -G video pi
# Logout and login again
```

### Performance Issues
**Problem**: Slow prediction
- Use only essential models (disable SVC if needed)
- Reduce camera resolution
- Close other applications

**Problem**: GUI freezing
- Check available RAM: `free -h`
- Restart Pi if memory is low
- Use lighter desktop environment

### Model Loading Issues
**Problem**: "No models loaded"
```bash
# Check model files exist
ls -la 02_Models/*/*.pkl

# Check file permissions
chmod 644 02_Models/*/*.pkl

# Test model loading
python3 -c "import pickle; pickle.load(open('02_Models/KNN/knn_classifier_model.pkl', 'rb'))"
```

## 📊 Performance Benchmarks

### Raspberry Pi 4 (4GB RAM)
- **Model Loading**: ~5-10 seconds
- **Image Capture**: Instant
- **Prediction Time**: 1-2 seconds
- **Memory Usage**: ~200MB
- **CPU Usage**: 15-25% during prediction

### Accuracy Comparison
| Model | Accuracy | Speed | Memory |
|-------|----------|-------|---------|
| Random Forest | 98.7% | Fast | Low |
| KNN | 95.2% | Medium | Low |
| SVC | 94.8% | Slow | Low |
| **Ensemble** | **98.7%** | **Fast** | **Low** |

## 🔄 Auto-Start Setup

### Method 1: Desktop Autostart
```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/arecanut.desktop << EOF
[Desktop Entry]
Type=Application
Name=Areca Pi Detector
Exec=python3 /home/pi/ArecaNut_GUI/Areca_Pi.py
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF
```

### Method 2: System Service
```bash
sudo nano /etc/systemd/system/arecanut.service
```

Add:
```ini
[Unit]
Description=Areca Nut Detection Service
After=graphical-session.target

[Service]
Type=simple
User=pi
Environment=DISPLAY=:0
ExecStart=/usr/bin/python3 /home/pi/ArecaNut_GUI/Areca_Pi.py
Restart=always

[Install]
WantedBy=graphical-session.target
```

Enable:
```bash
sudo systemctl enable arecanut.service
sudo systemctl start arecanut.service
```

## 📈 Field Deployment Tips

### 1. **Lighting Conditions**
- Use consistent lighting for best results
- Avoid direct sunlight or shadows
- Consider LED ring light for uniform illumination

### 2. **Camera Positioning**
- Mount camera 15-20cm from areca nuts
- Ensure stable mounting to reduce blur
- Use macro lens for close-up detail if needed

### 3. **Power Management**
- Use official Pi power adapter (5V 3A)
- Consider UPS/battery pack for field use
- Monitor temperature in enclosed cases

### 4. **Data Collection**
- Images auto-saved with timestamps
- Results can be logged to CSV file
- Consider cloud sync for data backup

## 🔮 Future Enhancements

### Planned Features
- **Batch processing** for multiple nuts
- **CSV export** of results
- **Audio feedback** using espeak
- **Web interface** for remote monitoring
- **Mobile app** companion

### Hardware Upgrades
- **Pi Camera Module** integration
- **Servo motor** for automated positioning
- **Load cell** for weight measurement
- **Environmental sensors** (temperature, humidity)

## 📞 Support

### Getting Help
1. **Check logs**: Run with `python3 Areca_Pi.py` in terminal
2. **Test components**: Use individual test scripts
3. **Update system**: `sudo apt update && sudo apt upgrade`
4. **Reinstall**: Re-run `setup_pi.sh`

### Common Commands
```bash
# Check system info
cat /proc/cpuinfo | grep Model

# Monitor resources
htop

# Test camera
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Check model files
find . -name "*.pkl" -ls
```

---

**Ready to detect areca nut ripeness on your Raspberry Pi!** 🥥🔍

*For technical details and model training, see the main project documentation.*