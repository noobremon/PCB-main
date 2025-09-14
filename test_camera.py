import cv2
import time
from camera_integration import OpenCVCamera

def test_camera():
    print("Testing camera connection...")
    
    # Initialize camera with default settings
    camera = OpenCVCamera(camera_index=0)
    
    # Connect to camera
    if not camera.connect():
        print("Failed to connect to camera")
        return
    
    print("Camera connected successfully!")
    print("Camera info:", camera.get_camera_info())
    
    try:
        # Try to capture and display 10 frames
        for i in range(10):
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
            
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        camera.disconnect()
        print("Camera disconnected")

if __name__ == "__main__":
    test_camera()
