import cv2

def test_camera():
    print("Testing camera with OpenCV...")
    
    # Try to open the default camera (index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow backend on Windows
    
    if not cap.isOpened():
        print("Error: Could not open camera with CAP_DSHOW, trying default backend...")
        cap = cv2.VideoCapture(0)  # Try default backend
    
    if not cap.isOpened():
        print("Error: Could not open camera with any backend")
        return
    
    print("Camera opened successfully!")
    print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    try:
        # Try to read a few frames
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame {i+1}")
                continue
                
            print(f"Captured frame {i+1} - Size: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imshow('Camera Test', frame)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    test_camera()
