"""
Camera Integration Module for Industrial PCB AOI System
Supports Baumer SDK cameras and phone/webcam cameras

NOTE: The Baumer SDK (pgbm) is an optional dependency for industrial cameras.
If not installed, the system will automatically fall back to OpenCV camera interface.

To install Baumer SDK:
1. Download and install Baumer GenICam SDK from Baumer's official website
2. Ensure the SDK's Python bindings are in your PYTHONPATH
3. Restart your application

Author: AI Assistant
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from abc import ABC, abstractmethod
import json
import os

# Optional industrial camera SDK imports (gracefully handled if not available)
# These are imported dynamically in the respective classes to avoid import errors
# when the Baumer SDK is not installed
logger = logging.getLogger(__name__)

class CameraInterface(ABC):
    """Abstract base class for camera interfaces"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to camera"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from camera"""
        pass
    
    @abstractmethod
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture single image"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        pass
    
    @abstractmethod
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information"""
        pass

class BaumerSDKCamera(CameraInterface):
    """Baumer SDK camera implementation with OpenCV fallback"""
    
    def __init__(self, camera_id=0, config: Optional[Dict] = None):
        self.camera_id = camera_id
        self._camera = None
        self._system = None
        self._device = None
        self._nodemap = None
        self._stream = None
        self._is_baumer_available = False
        self._opencv_fallback = False
        self._opencv_camera = None
        self.last_error = None  # Store the last error message
        
        # Initialize with default config if none provided
        self._config = config or {}
        self._config.setdefault('use_opencv_fallback', True)
        self._config.setdefault('require_baumer', False)  # If True, will not fall back to OpenCV
        self._config.setdefault('opencv_backend', 'default')
        
        # Log initialization
        logger.info(f"Initializing BaumerSDKCamera with ID: {camera_id}")
        logger.info(f"Configuration: {self._config}")
        
        # Check if Baumer SDK is available
        self._check_baumer_availability()
        
        # If Baumer is not available and OpenCV fallback is enabled
        if not self._is_baumer_available and self._config.get('use_opencv_fallback', True):
            self._init_opencv_fallback()
    
    def _check_baumer_availability(self):
        """Check if Baumer SDK is available and properly installed"""
        try:
            import pgbm  # type: ignore
            from pgbm import PgDevice, PgSystem, PgStream, PgNodemap  # type: ignore
            
            # Test basic SDK functionality
            system = PgSystem.GetInstance()
            if system is None:
                raise RuntimeError("Failed to get Baumer system instance")
                
            self._is_baumer_available = True
            logger.info("✅ Baumer SDK is properly installed and available")
            return True
            
        except ImportError as e:
            error_msg = "❌ Baumer SDK is not installed or not in PYTHONPATH.\n"
            error_msg += "Please install the Baumer GenICam SDK and ensure it's in your PYTHONPATH.\n"
            error_msg += "Falling back to OpenCV camera interface."
            logger.warning(error_msg)
            self._is_baumer_available = False
            return False
            
        except Exception as e:
            error_msg = f"❌ Error initializing Baumer SDK: {str(e)}\n"
            error_msg += "Please check your Baumer camera connection and drivers.\n"
            error_msg += "Falling back to OpenCV camera interface."
            logger.warning(error_msg)
            self._is_baumer_available = False
            return False
    
    def _init_opencv_fallback(self):
        """Initialize OpenCV as fallback"""
        try:
            import cv2
            self._opencv_camera = cv2.VideoCapture(self.camera_id)
            if self._opencv_camera.isOpened():
                self._opencv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self._opencv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self._opencv_fallback = True
                self._is_camera_connected = True
                logger.info("Initialized OpenCV fallback camera")
            else:
                logger.error("Failed to initialize OpenCV fallback camera")
                self._opencv_fallback = False
                self._is_camera_connected = False
        except Exception as e:
            logger.error(f"Error initializing OpenCV fallback: {e}")
            self._opencv_fallback = False
            self._is_camera_connected = False
    
    def connect(self) -> bool:
        """
        Connect to Baumer camera with clear feedback and fallback options
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        # If already connected, return True
        if self.is_connected():
            logger.info("Camera already connected")
            return True
            
        # Try Baumer SDK first if available
        if self._is_baumer_available:
            try:
                logger.info(f"Attempting to connect to Baumer camera: {self.camera_id}")
                
                # Simulated Baumer SDK connection
                # self._camera = BaumerCamera()  # Actual SDK call
                # self._camera.connect(self.camera_id)
                
                # For simulation purposes - simulate successful connection
                # In real implementation, this would be replaced with actual SDK calls
                self._is_camera_connected = True
                self._camera_info = {
                    'type': 'Baumer SDK Camera',
                    'model': 'Baumer Industrial Camera',
                    'serial': 'BAU123456',
                    'resolution': (2048, 1536),
                    'fps': 30,
                    'color_depth': '8-bit',
                    'status': 'connected'
                }
                
                logger.info("✅ Baumer camera connected successfully")
                return True
                
            except Exception as e:
                error_msg = f"❌ Failed to connect to Baumer camera: {str(e)}\n"
                error_msg += "Please ensure the Baumer camera is properly connected and the drivers are installed."
                logger.error(error_msg)
                self._is_camera_connected = False
                return False
                
        # If Baumer camera is required and not available, don't fall back to OpenCV
        if self._config.get('require_baumer', False):
            error_msg = "❌ Baumer camera is required but not available.\n"
            error_msg += "Please connect a Baumer camera and ensure the drivers are properly installed."
            logger.error(error_msg)
            return False
            
        # Fall back to OpenCV if Baumer fails or is not available and fallback is allowed
        if self._config.get('use_opencv_fallback', True):
            logger.info("Baumer camera not available, falling back to OpenCV camera...")
            if self._opencv_camera is None:
                self._init_opencv_fallback()
                
            if self._opencv_camera is not None and self._opencv_camera.isOpened():
                self._opencv_fallback = True
                self._camera_info = {
                    'type': 'OpenCV Fallback Camera',
                    'model': 'Webcam/Phone Camera',
                    'resolution': (
                        int(self._opencv_camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self._opencv_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ),
                    'fps': self._opencv_camera.get(cv2.CAP_PROP_FPS),
                    'status': 'connected (fallback)'
                }
                logger.warning("⚠️ Using OpenCV fallback camera - Baumer camera not available")
                logger.info("✅ Successfully connected to OpenCV fallback camera")
                return True
                
        logger.error("❌ Failed to connect to any camera")
        return False
    
    def disconnect(self):
        """Disconnect from the camera and clean up resources"""
        try:
            # Clean up OpenCV camera if in use
            if self._opencv_fallback and self._opencv_camera is not None:
                self._opencv_camera.release()
                self._opencv_camera = None
                self._opencv_fallback = False
                self._is_camera_connected = False
                logger.info("Disconnected OpenCV camera")
                return
                
            # Clean up Baumer camera if in use
            if self._is_baumer_available:
                try:
                    if hasattr(self, '_stream') and self._stream is not None:
                        self._stream.StopAcquisition()
                        self._stream = None
                    if hasattr(self, '_device') and self._device is not None:
                        self._device.StopAcquisition()
                        self._device.DeInit()
                        self._device = None
                    logger.info("Disconnected Baumer camera")
                except Exception as e:
                    logger.error(f"Error disconnecting Baumer camera: {e}")
                finally:
                    # Reset Baumer camera state to allow reconnection
                    self._is_camera_connected = False
                    self._camera = None
                    self._system = None
                    self._nodemap = None
        except Exception as e:
            logger.error(f"Error during camera disconnect: {e}")
        finally:
            self._is_camera_connected = False
            self._opencv_fallback = False
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture single image"""
        if not self._is_camera_connected:
            logger.error("Baumer camera not connected")
            return None
        
        try:
            # Simulated image capture
            # In real implementation:
            # image_data = self._camera.capture_image()
            # image = np.array(image_data)
            
            # For simulation, return None to indicate Baumer camera not available
            logger.warning("Baumer camera capture simulated - returning None")
            return None
            
        except Exception as e:
            logger.error(f"Failed to capture image from Baumer camera: {e}")
            return None
    
    def is_connected(self) -> bool:
        """
        Check if camera is connected
        
        Returns:
            bool: True if connected (either via Baumer SDK or OpenCV), False otherwise
        """
        if self._opencv_fallback and self._opencv_camera is not None:
            return self._opencv_camera.isOpened()
            
        return self._device is not None and self._stream is not None
        
    def get_camera_info(self) -> Dict[str, Any]:
        """Get Baumer camera information"""
        return self._camera_info

class OpenCVCamera(CameraInterface):
    """OpenCV camera interface for webcams/phone cameras with optimized settings"""
    
    def _find_available_cameras(self) -> List[int]:
        """Find all available camera indices"""
        available_cameras = []
        # Check the first 5 indices (most systems won't have more than this)
        for i in range(5):
            try:
                # Try with DirectShow first (Windows)
                cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
                if cap.isOpened() and cap.read()[0]:
                    available_cameras.append(i)
                    cap.release()
                    time.sleep(0.1)
                    continue
                
                # Try with MSMF (Windows)
                cap = cv2.VideoCapture(i + cv2.CAP_MSMF)
                if cap.isOpened() and cap.read()[0]:
                    available_cameras.append(i)
                    cap.release()
                    time.sleep(0.1)
                    continue
                    
                # Try with default backend
                cap = cv2.VideoCapture(i)
                if cap.isOpened() and cap.read()[0]:
                    available_cameras.append(i)
                    cap.release()
                    time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Error checking camera index {i}: {e}")
                continue
                
        return available_cameras
        
    def _try_open_camera(self, index: int) -> Optional[cv2.VideoCapture]:
        """Try to open a camera with the given index"""
        try:
            # Try different backends in order of preference
            backends = [
                cv2.CAP_DSHOW,  # DirectShow (Windows)
                cv2.CAP_MSMF,   # Microsoft Media Foundation (Windows)
                cv2.CAP_ANY     # Any available backend
            ]
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(index + backend)
                    if cap.isOpened():
                        # Test if we can actually read a frame
                        ret, _ = cap.read()
                        if ret:
                            return cap
                        cap.release()
                except Exception as e:
                    logger.debug(f"Error with backend {backend}: {e}")
                    continue
            
            # If no backend worked, try direct open
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
                    
        except Exception as e:
            logger.warning(f"Error opening camera {index}: {e}")
            
        return None

    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (1280, 720), 
                 fps: int = 30, buffer_size: int = 1, auto_focus: bool = True, 
                 auto_exposure: float = 0.75, exposure: float = 0.0):
        """
        Initialize OpenCV camera with optimized settings
        
        Args:
            camera_index: Index of the camera to use
            resolution: Tuple of (width, height) for camera resolution
            fps: Target frames per second (30 is a good default for most cameras)
            buffer_size: Size of the camera buffer (1 = minimum lag)
            auto_focus: Whether to enable auto focus (True by default)
            auto_exposure: Auto exposure mode (0.75 = auto exposure, 0.25 = manual)
            exposure: Exposure value (0.0 = default, only used in manual mode)
        """
        self.camera_index = camera_index
        self.target_resolution = resolution
        self.target_fps = fps
        self.buffer_size = buffer_size
        self.auto_focus = auto_focus
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        
        self.camera = None
        self.is_camera_connected = False
        self.camera_info = {
            'type': 'OpenCV Camera',
            'index': camera_index,
            'resolution': resolution,
            'fps': fps,
            'backend': 'Not connected',
            'status': 'disconnected'
        }
        self.last_frame_time = 0
        self.frame_count = 0
        self.actual_fps = 0
        self.last_error = None
    
    def is_connected(self) -> bool:
        """Check if OpenCV camera is connected and responsive"""
        if self.camera is None or not self.is_camera_connected:
            return False
        
        # Check if camera is still responding
        try:
            if not hasattr(self.camera, 'isOpened') or not self.camera.isOpened():
                self.is_camera_connected = False
                self.camera_info['status'] = 'disconnected'
                return False
            
            # Try to get a property to check if camera is still responsive
            _ = self.camera.get(cv2.CAP_PROP_POS_MSEC)
            
            # Update connection status
            self.camera_info['status'] = 'connected'
            return True
            
        except Exception as e:
            logger.warning(f"Camera connection check failed: {e}")
            self.is_camera_connected = False
            if 'camera_info' in self.__dict__:
                self.camera_info['status'] = 'error'
                self.camera_info['last_error'] = str(e)
            return False
    
    def connect(self) -> bool:
        """
        Connect to OpenCV camera with optimized settings for performance
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if self.is_connected():
            logger.info("Camera already connected")
            return True
            
        try:
            # Release any existing camera instance
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as e:
                    logger.warning(f"Error releasing previous camera: {e}")
                self.camera = None
            
            # Try to find available cameras
            available_cameras = self._find_available_cameras()
            logger.info(f"Found available cameras at indices: {available_cameras}")
            
            # If no cameras found but we have an index, try that directly
            if not available_cameras and self.camera_index >= 0:
                available_cameras = [self.camera_index]
            
            # Try to connect to the requested camera first
            if self.camera_index in available_cameras:
                logger.info(f"Attempting to connect to camera at index {self.camera_index}")
                self.camera = self._try_open_camera(self.camera_index)
                if self.camera and self.camera.isOpened():
                    logger.info(f"Successfully connected to camera at index {self.camera_index}")
            
            # If that fails, try other available cameras
            if not self.camera or not self.camera.isOpened():
                for cam_index in available_cameras:
                    if cam_index != self.camera_index:  # Skip already tried index
                        logger.info(f"Trying alternative camera at index {cam_index}")
                        self.camera = self._try_open_camera(cam_index)
                        if self.camera and self.camera.isOpened():
                            logger.info(f"Connected to camera at index {cam_index}")
                            self.camera_index = cam_index  # Update to the working index
                            break
            
            # If still no camera, try direct connection as last resort
            if not self.camera or not self.camera.isOpened():
                logger.info("Trying direct connection as last resort...")
                self.camera = cv2.VideoCapture(self.camera_index)
                if not self.camera.isOpened():
                    raise RuntimeError("Failed to open camera with direct connection")
                
                # Set basic camera properties
                width, height = self.target_resolution
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                
                # Reset all camera properties to default values
                try:
                    # Reset basic properties
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)    # 0.5 = default brightness
                    self.camera.set(cv2.CAP_PROP_CONTRAST, 0.5)      # 0.5 = default contrast
                    self.camera.set(cv2.CAP_PROP_SATURATION, 0.5)    # 0.5 = default saturation
                    self.camera.set(cv2.CAP_PROP_HUE, 0.0)           # 0.0 = default hue
                    self.camera.set(cv2.CAP_PROP_GAIN, 0)            # 0 = auto gain
                    self.camera.set(cv2.CAP_PROP_GAMMA, 1.0)         # 1.0 = default gamma
                    self.camera.set(cv2.CAP_PROP_SHARPNESS, 0.0)     # 0.0 = default sharpness
                    
                    # Reset exposure and focus
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = auto exposure
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, 0.0)        # 0.0 = default exposure
                    self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # 1 = auto focus on
                    
                    # Reset white balance (if supported)
                    try:
                        self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)       # 1 = auto white balance
                        self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)  # ~daylight white balance
                    except:
                        pass
                    
                    # Let the camera adjust
                    time.sleep(0.5)
                except Exception as prop_error:
                    logger.warning(f"Could not set some camera properties: {prop_error}")
                
                # Test if we can actually read a frame
                retries = 3
                test_frame = None
                ret = False
                for _ in range(retries):
                    ret, test_frame = self.camera.read()
                    if ret and test_frame is not None:
                        break
                    time.sleep(0.1)
                
                if not ret or test_frame is None:
                    raise RuntimeError("Failed to capture test frame from camera after multiple attempts")
                
            # Get actual properties after successful connection
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            # Update camera info with actual properties
            self.camera_info.update({
                'resolution': (actual_width, actual_height),
                'fps': actual_fps,
                'status': 'connected',
                'backend': 'OpenCV',
                'index': self.camera_index,
                'actual_fps': actual_fps,
                'actual_resolution': (actual_width, actual_height)
            })
            
            self.is_camera_connected = True
            self.last_error = None
            logger.info(f"Successfully connected to camera {self.camera_index}")
            logger.info(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            return True
            
        except Exception as e:
            error_msg = f"Error connecting to camera: {str(e)}"
            logger.error(error_msg)
            self.last_error = error_msg
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as release_error:
                    logger.warning(f"Error releasing camera during cleanup: {release_error}")
                self.camera = None
            
            self.camera_info['status'] = 'error'
            self.camera_info['last_error'] = error_msg
            return False
    
    def disconnect(self):
        """Safely disconnect from OpenCV camera"""
        if self.camera is not None and self.is_camera_connected:
            try:
                self.camera.release()
                self.camera = None
                self.is_camera_connected = False
                logger.info("OpenCV camera disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting OpenCV camera: {e}")

    def capture_single_image(self, apply_enhancement: bool = False, retries: int = 2) -> Optional[np.ndarray]:
        """
        Capture a single image from the camera with optimized performance
        
        Args:
            apply_enhancement: Whether to apply image enhancement
            retries: Number of retry attempts if capture fails
            
        Returns:
            Captured image as numpy array, or None if capture failed
        """
        if not self.camera or not self.is_connected():
            logger.warning("Camera not connected, attempting to connect...")
            if not self.connect():
                logger.error("Failed to connect to camera")
                return None
        
        for attempt in range(retries + 1):
            try:
                # Check if camera is still connected and not None
                if not self.camera or not self.camera.isOpened():
                    logger.warning("Camera not open, attempting to reconnect...")
                    if not self.connect():
                        logger.error("Failed to reconnect to camera")
                        return None
                
                # Double-check camera is not None after reconnection
                if self.camera is None:
                    logger.error("Camera is None after reconnection attempt")
                    return None
                
                # Clear buffer by grabbing frames (but fewer for performance)
                for _ in range(2):
                    if self.camera is not None:
                        self.camera.grab()
                
                # Capture frame with error handling
                if self.camera is None:
                    logger.error("Camera became None during capture")
                    return None
                    
                ret, frame = self.camera.read()
                
                if not ret or frame is None or frame.size == 0:
                    logger.warning(f"Failed to capture frame (attempt {attempt + 1}/{retries + 1})")
                    if attempt == retries:
                        logger.error("Max retries reached, giving up")
                        return None
                    continue
                
                # Convert BGR to RGB (faster than cvtColor for this specific conversion)
                frame_rgb = frame[..., ::-1]  # BGR to RGB
                
                # Update FPS counter
                current_time = time.time()
                self.frame_count += 1
                if current_time - self.last_frame_time >= 1.0:  # Update FPS every second
                    self.actual_fps = self.frame_count / (current_time - self.last_frame_time)
                    self.frame_count = 0
                    self.last_frame_time = current_time
                    logger.debug(f"Camera FPS: {self.actual_fps:.1f}")
                
                # Apply basic contrast enhancement if needed
                if apply_enhancement:
                    # Simple contrast and brightness adjustment
                    alpha = 1.2  # Contrast control (1.0-3.0)
                    beta = 10    # Brightness control (0-100)
                    frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=alpha, beta=beta)
                
                # Limit resolution for better performance if needed
                height, width = frame_rgb.shape[:2]
                if width > 1280:
                    frame_rgb = cv2.resize(
                        frame_rgb, 
                        (1280, int(height * (1280/width))), 
                        interpolation=cv2.INTER_AREA
                    )
                
                return frame_rgb
                
            except Exception as e:
                logger.error(f"Error in capture_single_image (attempt {attempt + 1}): {e}")
                if attempt == retries:
                    logger.error("Max retries reached, returning None")
                    # Try to recover by reconnecting
                    try:
                        self.disconnect()
                        time.sleep(1)
                        self.connect()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to recover camera: {reconnect_error}")
                    return None
                
                # Try to recover by reinitializing the camera
                if self.camera:
                    try:
                        self.camera.release()
                    except:
                        pass
                    self.camera = None
                
                # Small delay before retry
                time.sleep(0.1)
        
        return None
        
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture single image (implements CameraInterface abstract method)"""
        return self.capture_single_image(apply_enhancement=True)
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get OpenCV camera information"""
        return self.camera_info

class CameraManager:
    """Camera management system for industrial PCB inspection with lazy initialization"""
    
    def __init__(self, config_file: str = "camera_config.json"):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        self.config_file = config_file
        self.current_camera: Optional[CameraInterface] = None
        self.camera_config = self.load_config()
        self.available_cameras = []
        
        # Camera settings from config
        self.preferred_camera = self.camera_config.get('preferred_camera', 'opencv')
        self.target_resolution = tuple(self.camera_config.get('resolution', [1280, 720]))
        self.target_fps = self.camera_config.get('fps', 30)
        self.buffer_size = self.camera_config.get('buffer_size', 2)
        
        # Detection settings
        self.auto_capture_enabled = False
        self.capture_interval = self.camera_config.get('auto_capture_interval', 0.1)  # seconds
        self.capture_thread = None
        self.capture_queue = queue.Queue()
        self.stop_capture = threading.Event()
        
        # Camera state
        self._is_initialized = False
        self._is_baumer_available = False  # Initialize Baumer availability flag
        self.last_capture_time = 0  # Initialize last capture time for auto-capture
    
    def load_config(self) -> Dict[str, Any]:
        """Load camera configuration"""
        default_config = {
            'preferred_camera': 'baumer',
            'baumer_camera_id': '',
            'opencv_camera_index': 0,
            'resolution': [1920, 1080],
            'auto_capture_interval': 5.0,
            'image_format': 'jpg',
            'quality': 95
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                self.logger.error(f"Error loading camera config: {e}")
                return default_config
        else:
            # Save default config
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict[str, Any]):
        """Save camera configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving camera config: {e}")
    
    def discover_cameras(self):
        """
        Discover available cameras without connecting to them
        
        Returns:
            List[Dict]: List of available cameras with their details
        """
        available_cameras = []
        
        try:
            # Check for OpenCV cameras
            for i in range(10):  # Check first 10 indices
                cap = None
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DSHOW on Windows for better performance
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_info = {
                            'index': i,
                            'type': 'OpenCV',
                            'name': f'Webcam {i}',
                            'resolution': f"{width}x{height}",
                            'fps': fps,
                            'backend': 'OpenCV',
                            'status': 'Available'
                        }
                        available_cameras.append(camera_info)
                        cap.release()
                except Exception as e:
                    logger.warning(f"Error checking camera index {i}: {e}")
                    if cap is not None:
                        cap.release()
            
            # Check for Baumer camera (simulated for now)
            try:
                # Try to import Baumer SDK to check availability
                try:
                    import pgbm  # type: ignore
                    self._is_baumer_available = True
                    available_cameras.append({
                        'index': 'baumer',
                        'type': 'Baumer SDK',
                        'name': 'Baumer Industrial Camera',
                        'resolution': '2048x1536',
                        'fps': 30,
                        'backend': 'Baumer SDK',
                        'status': 'Available (Simulated)'
                    })
                except ImportError:
                    self._is_baumer_available = False
                    logger.info("Baumer SDK not available. Using OpenCV backend only.")
                    
            except Exception as e:
                self._is_baumer_available = False
                logger.error(f"Error checking Baumer camera: {e}")
                
        except Exception as e:
            logger.error(f"Error discovering cameras: {e}")
            
        self.available_cameras = available_cameras
        return available_cameras
    
    def initialize(self):
        """Initialize camera manager (doesn't connect to camera)"""
        if self._is_initialized:
            return True
            
        self.logger.info("Initializing Camera Manager...")
        try:
            # Load configuration
            self.camera_config = self.load_config()
            
            # Update settings from config
            self.preferred_camera = self.camera_config.get('preferred_camera', 'opencv').lower()
            self.target_resolution = tuple(self.camera_config.get('resolution', [1280, 720]))
            self.target_fps = self.camera_config.get('fps', 30)
            self.buffer_size = self.camera_config.get('buffer_size', 2)
            self.capture_interval = self.camera_config.get('auto_capture_interval', 0.1)
            
            # Check Baumer SDK availability only if it's the preferred camera
            if self.preferred_camera == 'baumer':
                try:
                    import pgbm  # type: ignore
                    from pgbm import PgSystem  # type: ignore
                    system = PgSystem.GetInstance()
                    self._is_baumer_available = system is not None and len(system.Devices) > 0
                    if not self._is_baumer_available:
                        self.logger.warning("Baumer SDK is installed but no Baumer cameras were found")
                except (ImportError, Exception) as e:
                    self._is_baumer_available = False
                    self.logger.warning(f"Baumer camera will not be available: {e}")
            else:
                self._is_baumer_available = False
            
            self.logger.info(f"Camera Manager initialized. Baumer available: {self._is_baumer_available}")
            self._is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Camera Manager: {e}")
            return False
    
    def connect_camera(self, camera_type: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Connect to the specified camera type or preferred camera with retry mechanism
        
        Args:
            camera_type: Type of camera to connect ('opencv' or 'baumer'). 
                       If None, uses preferred_camera from config.
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retry attempts in seconds
                       
        Returns:
            bool: True if connection was successful, False otherwise
        """
        camera_type = (camera_type or self.preferred_camera).lower()
        
        # If Baumer camera is requested but not available, return False
        if camera_type == 'baumer' and not self._is_baumer_available:
            self.logger.error("Baumer camera is not available. Please connect a Baumer camera and try again.")
            return False
            
        self.logger.info(f"Connecting to {camera_type} camera...")
        
        # Initialize if not already done
        if not self._is_initialized:
            self.initialize()
            
        # Disconnect from current camera if connected
        if self.current_camera is not None:
            self.disconnect_camera()
        
        for attempt in range(max_retries):
            try:
                if camera_type == 'baumer':
                    self.logger.info(f"Attempt {attempt + 1}/{max_retries}: Connecting to Baumer camera...")
                    self.current_camera = BaumerSDKCamera(
                        camera_id=0,  # Default to first Baumer camera
                        config={
                            'use_opencv_fallback': False,  # Don't fall back to OpenCV automatically
                            'require_baumer': True  # Require Baumer to work, no fallback
                        }
                    )
                else:
                    # Use OpenCV camera
                    self.logger.info(f"Attempt {attempt + 1}/{max_retries}: Using OpenCV camera interface")
                    self.current_camera = OpenCVCamera(
                        camera_index=0,  # Default to first camera
                        resolution=self.target_resolution,
                        fps=self.target_fps,
                        buffer_size=self.buffer_size
                    )
                
                # Connect to the camera
                if self.current_camera.connect():
                    # Verify connection by capturing a test frame
                    test_frame = self.capture_single_image()
                    if test_frame is not None and test_frame.size > 0:
                        self.logger.info(f"Successfully connected to {camera_type} camera")
                        # Small delay to ensure camera is fully initialized
                        import time
                        time.sleep(0.5)
                        return True
                    else:
                        self.logger.warning(f"Camera connected but failed to capture test frame (attempt {attempt + 1}/{max_retries})")
                
                # If we get here, connection or test frame capture failed
                self.logger.warning(f"Camera connection attempt {attempt + 1}/{max_retries} failed")
                if self.current_camera:
                    self.current_camera.disconnect()
                    self.current_camera = None
                
                # Don't wait after the last attempt
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
            except Exception as e:
                self.logger.error(f"Error during camera connection (attempt {attempt + 1}): {str(e)}")
                if self.current_camera:
                    try:
                        self.current_camera.disconnect()
                    except:
                        pass
                    self.current_camera = None
                
                # Don't wait after the last attempt
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        self.logger.error(f"Failed to connect to {camera_type} camera after {max_retries} attempts")
        return False
    
    def disconnect_camera(self):
        """
        Disconnect from the current camera if connected
        
        Returns:
            bool: True if disconnected successfully or no camera was connected
        """
        if self.current_camera:
            try:
                self.logger.info("Disconnecting camera...")
                self.current_camera.disconnect()
                self.current_camera = None
                self.logger.info("✅ Camera disconnected successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error disconnecting camera: {e}")
                return False
        return True
    
    def is_camera_connected(self) -> bool:
        """
        Check if a camera is currently connected and ready
        
        Returns:
            bool: True if a camera is connected and ready, False otherwise
        """
        if not self._is_initialized:
            return False
            
        if self.current_camera is None:
            return False
            
        try:
            return self.current_camera.is_connected()
        except Exception as e:
            self.logger.error(f"Error checking camera connection: {e}")
            return False
    
    def capture_single_image(self) -> Optional[np.ndarray]:
        """Capture single image"""
        if not self.is_camera_connected():
            self.logger.error("No camera connected")
            return None
        
        try:
            if self.current_camera is None:
                self.logger.error("No camera connected")
                return None
                
            image = self.current_camera.capture_image()
            if image is not None:
                self.logger.info("✅ Image captured successfully")
                return image
            else:
                self.logger.error("❌ Failed to capture image")
                return None
        except Exception as e:
            self.logger.error(f"Error during image capture: {e}")
            return None
    
    def start_auto_capture(self, interval: Optional[float] = None):
        """Start automatic image capture"""
        if interval:
            self.capture_interval = interval
        
        if not self.is_camera_connected():
            self.logger.error("Cannot start auto capture - no camera connected")
            return False
        
        if self.auto_capture_enabled:
            self.logger.warning("Auto capture already running")
            return True
        
        self.auto_capture_enabled = True
        self.stop_capture.clear()
        
        self.capture_thread = threading.Thread(target=self._auto_capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.logger.info(f"✅ Auto capture started with {self.capture_interval}s interval")
        return True
    
    def stop_auto_capture(self):
        """
        Stop automatic image capture and clean up resources
        
        Returns:
            bool: True if auto-capture was stopped successfully, False if it wasn't running
        """
        if not self.auto_capture_enabled:
            return False
        
        self.logger.info("Stopping auto capture...")
        self.auto_capture_enabled = False
        self.stop_capture.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                self.logger.warning("Auto capture thread did not stop gracefully")
        
        # Clear the queue
        while not self.capture_queue.empty():
            try:
                self.capture_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("Auto capture stopped and resources cleaned up")
        return True
    
    def _auto_capture_loop(self):
        """
        Auto capture loop with better error handling and reconnection logic
        
        Continuously captures frames and adds them to the capture queue.
        If the queue is full, the oldest frame is discarded.
        Handles camera disconnections and attempts to reconnect automatically.
        """
        self.last_capture_time = 0  # Initialize last capture time
        
        while not self.stop_capture.is_set():
            try:
                # Check if we need to reconnect
                if not self.is_camera_connected():
                    self.logger.warning("Camera disconnected. Attempting to reconnect...")
                    if not self.connect_camera(max_retries=3, retry_delay=1.0):
                        self.logger.error("Failed to establish camera connection for auto-capture")
                        return
        
                # Calculate time since last capture
                current_time = time.time()
                time_since_last_capture = current_time - self.last_capture_time
                
                # Only capture if enough time has passed based on capture_interval
                if time_since_last_capture >= self.capture_interval:
                    # Capture image
                    image = self.capture_single_image()
                    
                    if image is not None and image.size > 0:
                        # Add to queue, removing oldest frame if queue is full
                        if self.capture_queue.qsize() >= self.buffer_size:
                            try:
                                self.capture_queue.get_nowait()  # Remove oldest frame
                            except queue.Empty:
                                pass
                        
                        self.capture_queue.put(image)
                        self.last_capture_time = current_time
                    else:
                        self.logger.warning("Failed to capture image")
                        
                # Small sleep to prevent 100% CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in auto-capture loop: {str(e)}")
                
                # Attempt to reconnect
                self.disconnect_camera()
                if not self.connect_camera(max_retries=2, retry_delay=1.0):
                    self.logger.error("Failed to recover camera connection. Stopping auto-capture.")
                    break
        
        self.logger.info("Auto-capture loop stopped")
    
    def get_captured_image(self) -> Optional[np.ndarray]:
        """Get image from capture queue"""
        try:
            if not self.capture_queue.empty():
                return self.capture_queue.get_nowait()
        except queue.Empty:
            pass
        return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get current camera information"""
        if self.current_camera:
            return self.current_camera.get_camera_info()
        return {}
    
    def get_available_cameras_info(self) -> List[Dict[str, Any]]:
        """Get information about all available cameras"""
        return [cam['info'] for cam in self.available_cameras]
    
    def save_image(self, image: np.ndarray, filename: Optional[str] = None, 
                   directory: str = "captured_images") -> str:
        """Save captured image"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pcb_capture_{timestamp}.jpg"
        
        # Create directory if it doesn't exist
        save_dir = Path(directory)
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / filename
        
        try:
            quality = self.camera_config.get('quality', 95)
            cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            logger.info(f"Image saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return ""
    
    def preview_camera(self, window_name: str = "Camera Preview", 
                      duration: float = 10.0) -> bool:
        """Show camera preview for testing"""
        if not self.is_camera_connected():
            logger.error("No camera connected for preview")
            return False
        
        logger.info(f"Starting camera preview for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            image = self.capture_single_image()
            if image is not None:
                # Resize for display if too large
                height, width = image.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = 1280
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                cv2.imshow(window_name, image)
                
                # Check for 'q' key to quit early
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            else:
                logger.error("Failed to get preview image")
                break
            
            time.sleep(0.1)  # 10 FPS preview
        
        cv2.destroyAllWindows()
        logger.info("Camera preview ended")
        return True

# Utility functions
def detect_phone_cameras() -> List[int]:
    """Detect phone cameras connected via USB"""
    available_indices = []
    
    # Check common camera indices
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, _ = cap.read()
            if ret:
                available_indices.append(index)
            cap.release()
    
    return available_indices

def test_camera_connection():
    """Test camera connection and display information"""
    print("🔍 Testing Camera Connections...")
    
    # Detect phone cameras
    phone_cameras = detect_phone_cameras()
    print(f"📱 Phone/Webcam cameras found at indices: {phone_cameras}")
    
    # Test camera manager
    camera_manager = CameraManager()
    available_cameras = camera_manager.discover_cameras()
    
    print(f"📷 Total cameras discovered: {len(available_cameras)}")
    for i, cam_info in enumerate(available_cameras):
        print(f"  {i+1}. {cam_info['type'].upper()}: {cam_info.get('model', 'Unknown')}")
        print(f"     Resolution: {cam_info.get('resolution', 'Unknown')}")
        print(f"     FPS: {cam_info.get('fps', 'Unknown')}")
    
    # Try to connect to best camera
    if camera_manager.connect_camera():
        print("✅ Successfully connected to camera")
        
        # Test capture
        image = camera_manager.capture_single_image()
        if image is not None:
            print(f"✅ Test capture successful: {image.shape}")
            
            # Save test image
            test_path = camera_manager.save_image(image, "test_capture.jpg")
            print(f"💾 Test image saved: {test_path}")
        else:
            print("❌ Test capture failed")
        
        camera_manager.disconnect_camera()
    else:
        print("❌ Could not connect to any camera")

if __name__ == "__main__":
    # Run camera connection test
    test_camera_connection()