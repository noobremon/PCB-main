"""
Camera Diagnostic Tool
This script helps diagnose camera connection issues by testing different backends and settings.
"""
import cv2
import sys
import time
from pathlib import Path

def print_banner():
    print("\n" + "="*60)
    print("CAMERA DIAGNOSTIC TOOL".center(60))
    print("="*60)

def list_cameras():
    """Try to list all available cameras"""
    print("\n[1/4] Scanning for available cameras...")
    
    # Try different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default"),
    ]
    
    found_cameras = []
    
    for backend, name in backends:
        print(f"\nTrying {name} backend...")
        for i in range(0, 5):  # Check first 5 indices
            cap = cv2.VideoCapture(i + backend)
            if cap.isOpened():
                ret = cap.grab()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"  ✓ Camera {i}: {width}x{height} @ {fps:.1f} FPS")
                    found_cameras.append((i, backend, name))
                cap.release()
            else:
                print(f"  ✗ Camera {i}: Not available")
    
    return found_cameras

def test_camera(cam_index, backend):
    """Test a specific camera with the given backend"""
    print(f"\n[2/4] Testing camera {cam_index} with backend {backend}...")
    
    # Create capture object
    cap = cv2.VideoCapture(cam_index + backend)
    
    if not cap.isOpened():
        print(f"  ❌ Could not open camera {cam_index}")
        return False
    
    # Set some basic properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get and print camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Camera properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps:.1f}")
    
    # Test frame capture
    print("\n[3/4] Testing frame capture...")
    success = False
    
    for i in range(5):  # Try 5 times
        ret, frame = cap.read()
        if ret and frame is not None:
            success = True
            print(f"  ✓ Successfully captured frame {i+1}: {frame.shape[1]}x{frame.shape[0]}")
            
            # Save the first successful frame
            if i == 0:
                output_dir = Path("camera_test_output")
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f"camera_{cam_index}_test.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"  - Saved test image to: {filename}")
        else:
            print(f"  ❌ Failed to capture frame {i+1}")
    
    # Release resources
    cap.release()
    
    if success:
        print("\n[4/4] Test completed successfully! 🎉")
    else:
        print("\n[4/4] Test completed with errors ❌")
    
    return success

def main():
    print_banner()
    
    # List available cameras
    cameras = list_cameras()
    
    if not cameras:
        print("\n❌ No cameras found. Please check your camera connection and try again.")
        return
    
    print("\nFound the following cameras:")
    for i, (cam_idx, backend, backend_name) in enumerate(cameras):
        print(f"  {i+1}. Camera {cam_idx} (Backend: {backend_name})")
    
    # Test the first found camera by default
    if cameras:
        print("\nTesting the first available camera...")
        cam_idx, backend, backend_name = cameras[0]
        test_camera(cam_idx, backend)
    
    print("\nDiagnostic complete. Check the output above for any issues.")

if __name__ == "__main__":
    main()
