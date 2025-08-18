#!/usr/bin/env python3
"""
Simple camera test script for Raspberry Pi
Test your camera before running the main application
"""

import cv2
import os
from datetime import datetime

def test_camera():
    """Test camera functionality"""
    print("🥥 Areca Pi - Camera Test")
    print("=" * 30)
    
    # Try to initialize camera
    print("📷 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        print("💡 Troubleshooting:")
        print("   - Check USB connection")
        print("   - Try: lsusb | grep -i camera")
        print("   - Try: sudo usermod -a -G video $USER")
        return False
    
    print("✅ Camera initialized successfully!")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📊 Camera Settings:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print()
    
    print("🎥 Starting camera preview...")
    print("📝 Controls:")
    print("   - Press 'c' to capture image")
    print("   - Press 's' to save current frame")
    print("   - Press 'q' to quit")
    print()
    
    capture_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame!")
                break
            
            # Display frame
            cv2.imshow('Areca Pi Camera Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('s'):
                # Capture image
                capture_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_test_{timestamp}_{capture_count:03d}.jpg"
                
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"📸 Image saved: {filename}")
                else:
                    print(f"❌ Failed to save: {filename}")
                    
            elif key == ord('q'):
                print("👋 Exiting camera test...")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Camera test interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"📊 Test Summary:")
        print(f"   - Images captured: {capture_count}")
        print(f"   - Camera working: ✅")
        
        return True

def test_opencv():
    """Test OpenCV installation"""
    print("\n🔧 Testing OpenCV installation...")
    
    try:
        import cv2
        import numpy as np
        
        print(f"✅ OpenCV version: {cv2.__version__}")
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test basic operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = (0, 255, 0)  # Green image
        
        # Test image operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print("✅ Basic image operations working")
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV import error: {e}")
        return False
    except Exception as e:
        print(f"❌ OpenCV test error: {e}")
        return False

def check_system():
    """Check system information"""
    print("\n💻 System Information:")
    
    # Check if running on Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                # Extract Pi model
                for line in cpuinfo.split('\n'):
                    if 'Model' in line:
                        print(f"   - {line.strip()}")
                        break
            else:
                print("   - Not running on Raspberry Pi")
    except:
        print("   - Could not read CPU info")
    
    # Check camera devices
    print("\n📷 Camera Devices:")
    video_devices = []
    for i in range(5):  # Check /dev/video0 to /dev/video4
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    if video_devices:
        for device in video_devices:
            print(f"   - Found: {device}")
    else:
        print("   - No video devices found")
    
    # Check USB devices
    print("\n🔌 USB Devices:")
    try:
        import subprocess
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'camera' in line.lower() or 'webcam' in line.lower() or 'video' in line.lower():
                    print(f"   - {line.strip()}")
        else:
            print("   - Could not list USB devices")
    except:
        print("   - lsusb command not available")

def main():
    """Main test function"""
    print("🚀 Starting Areca Pi Camera Test Suite")
    print("=" * 50)
    
    # Check system
    check_system()
    
    # Test OpenCV
    opencv_ok = test_opencv()
    
    if not opencv_ok:
        print("\n❌ OpenCV test failed! Please install dependencies:")
        print("   sudo apt install -y python3-opencv")
        print("   pip3 install opencv-python")
        return
    
    # Test camera
    print("\n" + "=" * 50)
    camera_ok = test_camera()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    print(f"   - OpenCV: {'✅ OK' if opencv_ok else '❌ Failed'}")
    print(f"   - Camera: {'✅ OK' if camera_ok else '❌ Failed'}")
    
    if opencv_ok and camera_ok:
        print("\n🎉 All tests passed! Ready to run Areca_Pi.py")
        print("🚀 Next step: python3 Areca_Pi.py")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running main application.")
        print("💡 Run setup script: ./setup_pi.sh")

if __name__ == "__main__":
    main()