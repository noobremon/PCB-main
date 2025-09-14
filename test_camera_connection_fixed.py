"""
Test script to verify camera connection and capture functionality
"""
import cv2
import time
from camera_integration import CameraManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_camera_connection():
    """Test camera connection and capture functionality"""
    print("=== Testing Camera Connection ===")
    
    # Create camera manager
    camera_manager = CameraManager()
    
    try:
        # Discover available cameras
        print("\nDiscovering cameras...")
        cameras = camera_manager.discover_cameras()
        print(f"Found {len(cameras)} cameras:")
        for i, cam in enumerate(cameras):
            print(f"  {i+1}. {cam['name']} ({cam['type']}) - {cam['status']}")
        
        # Try to connect to the first available camera
        print("\nAttempting to connect to camera...")
        if camera_manager.connect_camera('opencv'):
            print("✅ Successfully connected to camera!")
            
            # Get camera info
            info = camera_manager.get_camera_info()
            print("\nCamera Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            # Test single capture
            print("\nTesting single image capture...")
            for i in range(3):
                print(f"Capture attempt {i+1}...")
                try:
                    frame = camera_manager.capture_single_image()
                    if frame is not None:
                        print(f"✅ Success! Captured image shape: {frame.shape}")
                        # Save the captured image
                        cv2.imwrite(f'test_capture_{i}.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        print(f"  - Saved as 'test_capture_{i}.jpg'")
                    else:
                        print("❌ Failed to capture image (returned None)")
                except Exception as e:
                    print(f"❌ Error during capture: {str(e)}")
                
                time.sleep(1)  # Small delay between captures
            
            # Test auto-capture
            print("\nTesting auto-capture mode (3 seconds)...")
            camera_manager.start_auto_capture(interval=0.5)  # Capture every 500ms
            
            start_time = time.time()
            while time.time() - start_time < 3:  # Run for 3 seconds
                frame = camera_manager.get_captured_image()
                if frame is not None:
                    print(f"  - Captured frame: {frame.shape}")
                time.sleep(0.1)
                
            camera_manager.stop_auto_capture()
            print("✅ Auto-capture test completed")
            
        else:
            print("❌ Failed to connect to any camera")
            
    except Exception as e:
        print(f"❌ Error during camera test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("\nCleaning up...")
        camera_manager.disconnect_camera()
        print("Test completed!")

if __name__ == "__main__":
    test_camera_connection()
