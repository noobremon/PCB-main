import cv2
import numpy as np
import time

def test_camera(camera_index=0, num_frames=10):
    print(f"Testing camera {camera_index}...")
    
    # Try different backends
    backends = [
        (cv2.CAP_DSHOW, 'DirectShow'),
        (cv2.CAP_MSMF, 'Media Foundation'),
        (cv2.CAP_ANY, 'Default')
    ]
    
    for backend, name in backends:
        print(f"\nTrying {name} backend...")
        cap = cv2.VideoCapture(camera_index + backend)
        
        if not cap.isOpened():
            print(f"  ❌ Could not open camera with {name}")
            cap.release()
            continue
            
        print(f"  ✅ Camera opened with {name}")
        
        # Try to set some properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Resolution: {width}x{height}, FPS: {fps}")
        
        # Try to capture frames
        success_count = 0
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print(f"  ❌ Failed to capture frame {i+1}")
                continue
                
            success_count += 1
            print(f"  ✅ Captured frame {i+1}: {frame.shape[1]}x{frame.shape[0]}")
            
            # Display the frame
            cv2.imshow('Camera Test', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
        print(f"  Successfully captured {success_count}/{num_frames} frames")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        if success_count > 0:
            print("\n✅ Camera test completed successfully!")
            return True
    
    print("\n❌ Could not capture any frames with any backend")
    return False

if __name__ == "__main__":
    test_camera()
