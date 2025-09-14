import cv2
import time
import logging
from camera_integration import OpenCVCamera

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_camera():
    print("Testing camera connection...")
    
    camera = None  # Initialize to None to avoid unbound variable warning
    try:
        # Initialize camera with default settings
        print("Initializing camera...")
        camera = OpenCVCamera(camera_index=0)
        
        # Connect to camera
        print("Attempting to connect to camera...")
        if not camera.connect():
            print("Failed to connect to camera")
            return
        
        print("Camera connected successfully!")
        print("Camera info:", camera.get_camera_info())
        
        try:
            # Try to capture and display 10 frames
            for i in range(10):
                print(f"\n--- Frame {i+1} ---")
                frame = camera.capture_single_image(apply_enhancement=True)
                if frame is not None:
                    print(f"Captured frame {i+1}/10 - Size: {frame.shape[1]}x{frame.shape[0]}")
                    # Display the frame
                    cv2.imshow('Camera Test', frame)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                else:
                    print(f"Failed to capture frame {i+1}")
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error during frame capture: {e}")
            
    except Exception as e:
        print(f"Error initializing camera: {e}")
        
    finally:
        # Clean up
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        if camera is not None:
            camera.disconnect()
        print("Test completed")

if __name__ == "__main__":
    test_camera()
