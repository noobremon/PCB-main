"""
Test script to verify camera connection with the updated OpenCVCamera class
"""
import cv2
import logging
import sys
from camera_integration import OpenCVCamera

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def test_camera_connection(camera_index=0):
    """Test camera connection and display basic info"""
    print("\n=== Testing Camera Connection ===")
    print(f"Attempting to connect to camera at index {camera_index}")
    
    # Create camera instance with default settings
    camera = OpenCVCamera(
        camera_index=camera_index,
        resolution=(1280, 720),
        fps=30,
        buffer_size=1
    )
    
    # Try to connect
    if camera.connect():
        print("\n✅ Camera connected successfully!")
        print("\n=== Camera Information ===")
        info = camera.get_camera_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Try to capture a frame
        print("\n=== Testing Frame Capture ===")
        frame = camera.capture_image()
        if frame is not None:
            print(f"✅ Successfully captured frame: {frame.shape[1]}x{frame.shape[0]}")
            
            # Save test image
            import os
            os.makedirs("test_output", exist_ok=True)
            output_path = os.path.join("test_output", "test_capture.jpg")
            cv2.imwrite(output_path, frame)
            print(f"✅ Test frame saved to: {os.path.abspath(output_path)}")
        else:
            print("❌ Failed to capture frame")
            
        # Clean up
        camera.disconnect()
        print("\n✅ Camera disconnected")
    else:
        print("\n❌ Failed to connect to camera")
        if hasattr(camera, 'last_error'):
            print(f"Error: {camera.last_error}")

if __name__ == "__main__":
    setup_logging()
    
    # Test default camera (index 0)
    test_camera_connection(0)
    
    # If you have multiple cameras, you can test them by uncommenting:
    # test_camera_connection(1)
    # test_camera_connection(2)
