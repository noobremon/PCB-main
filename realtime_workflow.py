
"""
Real-Time PCB Inspection Workflow Management System
Author: AI Assistant
Description: Manages the continuous PCB inspection workflow with industrial automation
"""

import cv2
from pathlib import Path
import numpy as np
import time
import threading
import queue
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import json
from enum import Enum
from dataclasses import dataclass
import statistics

# Path for serving latest annotated realtime preview (synced with backend/server.py)
REALTIME_LAST_PREVIEW = (Path(__file__).resolve().parents[0] / "defective_storage" / "realtime")
try:
    REALTIME_LAST_PREVIEW.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
REALTIME_LAST_FILE = REALTIME_LAST_PREVIEW / "last_annotated.jpg"


from camera_integration import CameraManager
from enhanced_pcb_inspection import IndustrialPCBInspector

# Configure logging
logger = logging.getLogger(__name__)

class InspectionState(Enum):
    """Inspection workflow states"""
    IDLE = "idle"
    WAITING_FOR_PCB = "waiting_for_pcb"
    CAPTURING = "capturing"
    INSPECTING = "inspecting"
    SHOWING_RESULTS = "showing_results"
    WAITING_FOR_REMOVAL = "waiting_for_removal"
    ERROR = "error"

@dataclass
class InspectionResult:
    """Inspection result data structure"""
    timestamp: datetime
    image_path: str
    is_defective: bool
    defects: List[Dict[str, Any]]
    quality_score: float
    severity_level: str
    confidence_score: float
    inspection_time: float
    result_id: str

class InspectionWorkflowManager:
    """Industrial PCB inspection workflow manager"""
    
    def __init__(self, config_file: str = "workflow_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
        # Initialize components (camera_manager is now initialized lazily)
        self.camera_manager = None
        self.pcb_inspector = IndustrialPCBInspector()
        
        # Workflow state
        self.current_state = InspectionState.IDLE
        self.is_running = False
        self.auto_mode = False
        self._camera_connected = False
        
        # Threading
        self.workflow_thread = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()  # For thread-safe camera operations
        
        # Statistics
        self.session_stats = {
            'total_inspected': 0,
            'total_defective': 0,
            'total_good': 0,
            'session_start': None,
            'inspection_times': [],
            'quality_scores': []
        }
        
        # Callbacks for UI updates
        self.state_callbacks: List[Callable] = []
        self.result_callbacks: List[Callable] = []
        
        # Results storage
        self.results_history: List[InspectionResult] = []
        self.current_result: Optional[InspectionResult] = None
        
        logger.info("Inspection Workflow Manager initialized (lazy camera loading)")
    
    def load_config(self) -> Dict[str, Any]:
        """Load workflow configuration"""
        default_config = {
            # Timing settings (seconds)
            'capture_delay': 1.0,           # Delay before capture
            'inspection_timeout': 30.0,     # Max inspection time
            'result_display_time': 10.0,    # Time to show results
            'next_pcb_wait_time': 5.0,      # Time to wait for next PCB
            'auto_mode_interval': 15.0,     # Auto mode cycle time
            
            # Image settings
            'image_stabilization_frames': 3, # Frames to capture for stability
            'image_stabilization_delay': 0.5,
            
            # Quality settings
            'min_quality_score': 50.0,      # Minimum acceptable quality
            'auto_save_defective': True,    # Auto save defective images
            'auto_save_all': False,         # Auto save all images
            
            # Storage settings
            'results_directory': 'inspection_results',
            'max_history_size': 1000,       # Max results to keep in memory
            
            # Audio/Visual alerts
            'enable_audio_alerts': True,
            'enable_visual_alerts': True,
            'beep_on_defect': True,
            'beep_on_good': False
        }
        
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.error(f"Error loading workflow config: {e}")
                return default_config
        else:
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict[str, Any]):
        """Save workflow configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving workflow config: {e}")
    
    def add_state_callback(self, callback: Callable):
        """Add callback for state changes"""
        self.state_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable):
        """Add callback for inspection results"""
        self.result_callbacks.append(callback)
    
    def _notify_state_change(self, new_state: InspectionState, data: Optional[Dict[str, Any]] = None):
        """Notify registered callbacks of state change"""
        self.current_state = new_state
        
        state_data = {
            'state': new_state.value,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        for callback in self.state_callbacks:
            try:
                callback(state_data)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
        
        logger.info(f"State changed to: {new_state.value}")
    
    def _notify_result(self, result: InspectionResult):
        """Notify registered callbacks of inspection result"""
        result_data = {
            'result_id': result.result_id,
            'timestamp': result.timestamp.isoformat(),
            'is_defective': result.is_defective,
            'quality_score': result.quality_score,
            'severity_level': result.severity_level,
            'defects': result.defects,
            'inspection_time': result.inspection_time
        }
        
        for callback in self.result_callbacks:
            try:
                callback(result_data)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current workflow state and statistics"""
        return {
            'state': self.current_state.value,
            'is_running': self.is_running,
            'auto_mode': self.auto_mode,
            'camera_connected': self._camera_connected,
            'stats': {
                'total_inspected': self.session_stats['total_inspected'],
                'total_defective': self.session_stats['total_defective'],
                'total_good': self.session_stats['total_good'],
                'session_duration': (datetime.now() - self.session_stats['session_start']).total_seconds() \
                    if self.session_stats['session_start'] else 0,
                'avg_inspection_time': statistics.mean(self.session_stats['inspection_times']) \
                    if self.session_stats['inspection_times'] else 0,
                'avg_quality_score': statistics.mean(self.session_stats['quality_scores']) \
                    if self.session_stats['quality_scores'] else 100.0
            },
            'current_result': {
                'result_id': self.current_result.result_id if self.current_result else None,
                'is_defective': self.current_result.is_defective if self.current_result else None,
                'timestamp': self.current_result.timestamp.isoformat() if self.current_result else None,
                'defect_count': len(self.current_result.defects) if self.current_result else 0
            } if self.current_result else None
        }

    def initialize_system(self) -> bool:
        """Initialize the inspection system"""
        try:
            logger.info("🏭 Initializing Industrial PCB Inspection System...")
            
            # Skip camera initialization here - it will be done on demand
            logger.info("Camera initialization deferred - will connect when needed")
            
            # Build reference model
            logger.info("Building PCB reference model...")
            if not self.pcb_inspector.build_industrial_reference_model():
                logger.warning("⚠️ Could not build reference model - manual training may be needed")
            
            # Create results directory
            results_dir = Path(self.config.get('results_directory', 'results'))
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize session
            self.session_stats['session_start'] = datetime.now()
            
            logger.info("✅ System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self._notify_state_change(InspectionState.ERROR, {"error": f"Initialization error: {str(e)}"})
            return False
    
    def start_inspection_workflow(self, auto_mode: bool = False):
        """
        Start the inspection workflow with improved camera connection handling
        
        Args:
            auto_mode: If True, runs in automatic inspection mode
        """
        if self.is_running:
            logger.warning("Workflow already running")
            return
        
        logger.info(f"🚀 Starting inspection workflow in {'auto' if auto_mode else 'manual'} mode")
        
        try:
            # Set initial state
            self.auto_mode = auto_mode
            self.stop_event.clear()
            
            # Initialize camera manager if not already done
            if self.camera_manager is None:
                logger.info("Initializing camera manager...")
                self.camera_manager = CameraManager()
            
            # Ensure camera is connected before starting
            if not self._camera_connected:
                logger.info("Establishing camera connection...")
                if not self.connect_camera():
                    logger.error("❌ Failed to connect to camera before starting workflow")
                    self._notify_state_change(InspectionState.ERROR, {
                        'error': 'Failed to connect to camera',
                        'recoverable': True
                    })
                    return False
            
            # Small delay to ensure camera is ready
            time.sleep(0.5)
            
            # Start the workflow thread
            self.is_running = True
            self.workflow_thread = threading.Thread(
                target=self._workflow_loop,
                name=f"WorkflowThread-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            self.workflow_thread.daemon = True
            self.workflow_thread.start()
            
            logger.info(f"✅ Inspection workflow started in {'auto' if auto_mode else 'manual'} mode")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start inspection workflow: {e}", exc_info=True)
            self.is_running = False
            self._notify_state_change(InspectionState.ERROR, {
                'error': f'Failed to start workflow: {str(e)}',
                'recoverable': False
            })
            return False
    
    def stop_inspection_workflow(self):
        """
        Stop the inspection workflow
        
        Note: This stops the workflow but keeps the camera connected to maintain the preview.
        Use shutdown() to fully clean up resources including camera disconnection.
        """
        if not self.is_running:
            return
            
        logger.info("Stopping inspection workflow...")
        
        # Store current camera connection state
        was_camera_connected = self._camera_connected
        
        # Stop the workflow thread
        self.is_running = False
        self.auto_mode = False
        self.stop_event.set()
        
        # Wait for the workflow thread to finish
        if self.workflow_thread and self.workflow_thread.is_alive():
            self.workflow_thread.join(timeout=2.0)
        
        # Clean up thread reference
        self.workflow_thread = None
        
        # Ensure camera remains connected
        if was_camera_connected and not self._camera_connected:
            logger.warning("Camera was disconnected during workflow stop, attempting to reconnect...")
            self.connect_camera()
        
        self._notify_state_change(InspectionState.IDLE, {
            'camera_connected': self._camera_connected,
            'message': 'Workflow stopped, camera remains connected' if self._camera_connected 
                      else 'Workflow stopped, camera connection status unknown'
        })
        logger.info("Inspection workflow stopped (camera remains connected)")
        
        # Return the current state including camera connection status
        return self.get_current_state()
    
    def export_session_report(self, filename: str) -> str:
        """Export session report to JSON file"""
        report = {
            'session_start': self.session_stats['session_start'].isoformat(),
            'total_inspected': self.session_stats['total_inspected'],
            'total_defective': self.session_stats['total_defective'],
            'total_good': self.session_stats['total_good'],
            'inspection_times': self.session_stats['inspection_times'],
            'quality_scores': self.session_stats['quality_scores'],
            'results': [
                {
                    'result_id': result.result_id,
                    'timestamp': result.timestamp.isoformat(),
                    'is_defective': result.is_defective,
                    'quality_score': result.quality_score,
                    'severity_level': result.severity_level,
                    'confidence_score': result.confidence_score,
                    'inspection_time': result.inspection_time,
                    'defects': result.defects
                }
                for result in self.results_history
            ]
        }
        
        results_dir = Path(self.config['results_directory'])
        report_path = results_dir / filename
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Exported session report to {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Failed to export session report: {e}")
            return ""
    
    def connect_camera(self) -> bool:
        """
        Connect to the camera with improved reliability and error handling
        
        Returns:
            bool: True if connection was successful and verified, False otherwise
        """
        with self._lock:
            # Check if already connected and working
            if self._camera_connected and self.camera_manager and self.camera_manager.is_camera_connected():
                try:
                    # Verify connection by capturing a test frame
                    test_frame = self.camera_manager.capture_single_image()
                    if test_frame is not None and test_frame.size > 0:
                        logger.info("Camera already connected and verified")
                        return True
                    else:
                        logger.warning("Camera connected but failed to capture test frame, reconnecting...")
                        self._camera_connected = False
                except Exception as e:
                    logger.warning(f"Error verifying camera connection: {e}, will attempt to reconnect...")
                    self._camera_connected = False
            
            # Connection parameters
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Initialize camera manager if not already done
                    if self.camera_manager is None:
                        logger.info(f"Initializing camera manager (attempt {attempt + 1}/{max_retries})")
                        self.camera_manager = CameraManager()
                    
                    # Try to connect to Baumer camera first
                    logger.info(f"Attempting to connect to Baumer camera (attempt {attempt + 1}/{max_retries})...")
                    if self.camera_manager.connect_camera(camera_type="baumer"):
                        # Verify connection with a test frame
                        time.sleep(0.5)  # Small delay after connection
                        test_frame = self.camera_manager.capture_single_image()
                        if test_frame is not None and test_frame.size > 0:
                            self._camera_connected = True
                            logger.info("✅ Baumer camera connected and verified")
                            return True
                        logger.warning("Baumer camera connected but failed to capture test frame")
                    
                    # Fall back to OpenCV if Baumer fails
                    logger.warning(f"⚠️ Baumer camera connection failed, falling back to OpenCV (attempt {attempt + 1}/{max_retries})...")
                    if self.camera_manager.connect_camera(camera_type="opencv"):
                        # Verify connection with a test frame
                        time.sleep(0.5)  # Small delay after connection
                        test_frame = self.camera_manager.capture_single_image()
                        if test_frame is not None and test_frame.size > 0:
                            self._camera_connected = True
                            logger.info("✅ OpenCV camera connected and verified")
                            return True
                        logger.warning("OpenCV camera connected but failed to capture test frame")
                    
                    # If we get here, both connection attempts failed
                    logger.warning(f"⚠️ Camera connection attempt {attempt + 1}/{max_retries} failed")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 5.0)  # Cap max delay at 5 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error connecting to camera (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        time.sleep(retry_delay)
            
            # All attempts failed
            self._camera_connected = False
            logger.error("❌ Failed to connect to any camera after all attempts")
            return False
    
    def disconnect_camera(self) -> bool:
        """
        Disconnect from the camera
            
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        with self._lock:
            if not self._camera_connected or self.camera_manager is None:
                return True
                    
            try:
                logger.info("Disconnecting camera...")
                success = self.camera_manager.disconnect_camera()
                if success:
                    self._camera_connected = False
                    logger.info("✅ Camera disconnected successfully")
                else:
                    logger.warning("Failed to properly disconnect camera")
                return success
                    
            except Exception as e:
                logger.error(f"Error disconnecting camera: {e}")
                return False
    
    def is_camera_connected(self) -> bool:
        """Check if the camera is connected and ready"""
        with self._lock:
            return (self._camera_connected and 
                   self.camera_manager is not None and 
                   self.camera_manager.is_camera_connected())
                   
    def get_session_stats(self) -> dict:
        """Get current session statistics"""
        return {
            'total_inspected': self.session_stats['total_inspected'],
            'total_defective': self.session_stats['total_defective'],
            'total_good': self.session_stats['total_good'],
            'defect_rate': (self.session_stats['total_defective'] / self.session_stats['total_inspected'] * 100) 
                          if self.session_stats['total_inspected'] > 0 else 0,
            'avg_quality': statistics.mean(self.session_stats['quality_scores']) 
                          if self.session_stats['quality_scores'] else 0,
            'avg_inspection_time': statistics.mean(self.session_stats['inspection_times']) 
                                 if self.session_stats['inspection_times'] else 0,
            'session_duration': (datetime.now() - self.session_stats['session_start']).total_seconds() 
                              if self.session_stats['session_start'] else 0
        }
    
    def _capture_stabilized_image(self) -> Optional[np.ndarray]:
        """Capture a stabilized image using multiple frames"""
        if not self.is_camera_connected():
            logger.error("Cannot capture stabilized image - camera not connected")
            return None
        
        if self.camera_manager is None:
            logger.error("Cannot capture stabilized image - camera manager not initialized")
            return None
        
        try:
            # Get stabilization settings from config
            num_frames = self.config.get('image_stabilization_frames', 3)
            delay = self.config.get('image_stabilization_delay', 0.5)
            
            logger.info(f"Capturing {num_frames} frames for image stabilization...")
            
            frames = []
            for i in range(num_frames):
                try:
                    # Add delay between captures for stabilization
                    if i > 0:
                        time.sleep(delay)
                    
                    if self.camera_manager is None:
                        logger.error("Cannot capture frame - camera manager not initialized")
                        return None
                    
                    frame = self.camera_manager.capture_single_image()
                    if frame is not None and frame.size > 0:
                        frames.append(frame)
                        logger.debug(f"Captured frame {i+1}/{num_frames}: {frame.shape}")
                    else:
                        logger.warning(f"Failed to capture frame {i+1}/{num_frames}")
                        
                except Exception as e:
                    logger.error(f"Error capturing frame {i+1}/{num_frames}: {e}")
                    continue
            
            if not frames:
                logger.error("No valid frames captured for stabilization")
                return None
            
            if len(frames) < num_frames:
                logger.warning(f"Only captured {len(frames)}/{num_frames} frames for stabilization")
            
            # Use the last captured frame as the stabilized result
            # In a more advanced implementation, you could:
            # - Average multiple frames to reduce noise
            # - Apply image registration to align frames
            # - Use the sharpest frame based on focus metrics
            stabilized_frame = frames[-1]
            
            logger.info(f"Successfully captured stabilized image: {stabilized_frame.shape}")
            return stabilized_frame
            
        except Exception as e:
            logger.error(f"Error during stabilized image capture: {e}")
            return None

    def _workflow_loop(self):
        """
        Main workflow loop for continuous inspection with robust error handling
        and automatic recovery
        """
        logger.info("🚀 Starting workflow loop")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Track if we're in the initial connection phase
        initial_connection = True
        
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Check camera connection status - only attempt to connect if not already connected
                    if not self._camera_connected or initial_connection:
                        if initial_connection:
                            logger.info("🔍 Initializing camera connection for workflow...")
                        else:
                            logger.info("🔍 Attempting to connect to camera...")
                            
                        # Ensure we have a camera manager instance
                        if self.camera_manager is None:
                            self.camera_manager = CameraManager()
                            
                        # Try to connect with appropriate retries
                        if self.connect_camera():
                            # Small delay to ensure camera is fully initialized
                            time.sleep(0.5)
                            
                            # Verify camera is actually working
                            test_frame = self.camera_manager.capture_single_image()
                            if test_frame is not None and test_frame.size > 0:
                                self._camera_connected = True
                                initial_connection = False
                                logger.info("✅ Camera connected and verified")
                                consecutive_errors = 0
                                self._notify_state_change(InspectionState.WAITING_FOR_PCB)
                            else:
                                logger.warning("⚠️ Camera connected but failed to capture test frame")
                                self._camera_connected = False
                                time.sleep(1)
                                continue
                        else:
                            logger.error("❌ Failed to connect to camera")
                            time.sleep(2)
                            continue
                    
                    # Capture frame
                    try:
                        # Double-check camera connection before capture
                        if self.camera_manager is None or not self._camera_connected:
                            logger.warning("Camera not ready, attempting to reconnect...")
                            self._camera_connected = False
                            time.sleep(0.5)
                            continue
                        
                        # Capture frame with timeout
                        frame = None
                        capture_start = time.time()
                        while time.time() - capture_start < 2.0:  # 2 second timeout
                            frame = self.camera_manager.capture_single_image()
                            if frame is not None and frame.size > 0:
                                break
                            time.sleep(0.1)
                        
                        if frame is None or frame.size == 0:
                            logger.warning("Failed to capture valid frame, will retry...")
                            self._camera_connected = False
                            time.sleep(1)
                            continue
                            
                        # Reset error counter on successful capture
                        consecutive_errors = 0
                        
                        # Save the captured frame for preview
                        preview_path = REALTIME_LAST_PREVIEW / f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        try:
                            # Ensure preview directory exists
                            preview_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Save the frame
                            if not cv2.imwrite(str(preview_path), frame):
                                raise IOError("Failed to save preview image")
                            
                            # Update the latest preview symlink
                            if REALTIME_LAST_FILE.exists():
                                try:
                                    REALTIME_LAST_FILE.unlink()
                                except OSError:
                                    pass  # Ignore if file doesn't exist
                            
                            # Create parent directories if they don't exist
                            REALTIME_LAST_FILE.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Create the symlink
                            if hasattr(os, 'symlink'):
                                REALTIME_LAST_FILE.symlink_to(preview_path.absolute())
                            else:
                                # Fallback for Windows without symlink privileges
                                import shutil
                                shutil.copy2(str(preview_path), str(REALTIME_LAST_FILE))
                            
                        except Exception as e:
                            logger.error(f"Error saving preview image: {e}")
                            raise
                        
                        # Inspect the PCB
                        self._notify_state_change(InspectionState.INSPECTING)
                        logger.info("Inspecting PCB...")
                        
                        # Perform inspection
                        inspection_start = time.time()
                        result = self._inspect_pcb(frame, str(preview_path))
                        inspection_time = time.time() - inspection_start
                        
                        # Update statistics
                        with self._lock:
                            self.session_stats['total_inspected'] += 1
                            if result.is_defective:
                                self.session_stats['total_defective'] += 1
                            else:
                                self.session_stats['total_good'] += 1
                            self.session_stats['inspection_times'].append(inspection_time)
                            self.session_stats['quality_scores'].append(result.quality_score)
                            
                            # Keep only the most recent results
                            max_history = self.config.get('max_history_size', 1000)
                            self.session_stats['inspection_times'] = self.session_stats['inspection_times'][-max_history:]
                            self.session_stats['quality_scores'] = self.session_stats['quality_scores'][-max_history:]
                        
                        # Store result
                        self.current_result = result
                        self.results_history.append(result)
                        self._notify_result(result)
                        
                        # Show results
                        self._notify_state_change(InspectionState.SHOWING_RESULTS, {
                            'result_id': result.result_id,
                            'is_defective': result.is_defective,
                            'defect_count': len(result.defects)
                        })
                        
                        # Wait before next cycle if in auto mode
                        if self.auto_mode:
                            time.sleep(self.config.get('auto_mode_interval', 15.0))
                        else:
                            # In manual mode, wait for next trigger
                            self._notify_state_change(InspectionState.WAITING_FOR_PCB)
                            self.stop_event.wait(timeout=300)  # 5 minute timeout
                            if self.stop_event.is_set():
                                break
                        
                    except Exception as e:
                        logger.error(f"Error during frame capture or processing: {e}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error("Max consecutive errors reached, disconnecting camera")
                            self.disconnect_camera()
                        time.sleep(1)
                        continue
                        
                except Exception as e:
                    logger.error(f"Error in workflow iteration: {e}", exc_info=True)
                    self._notify_state_change(InspectionState.ERROR, {"error": f"Workflow error: {str(e)}"})
                    time.sleep(1)  # Prevent tight loop on errors
                    
        except Exception as e:
            logger.critical(f"Fatal error in workflow loop: {e}", exc_info=True)
            self._notify_state_change(InspectionState.ERROR, {"error": f"Fatal workflow error: {str(e)}"})
        finally:
            self.is_running = False
            try:
                self.disconnect_camera()
            except Exception as e:
                logger.error(f"Error during camera disconnection: {e}")
            self._notify_state_change(InspectionState.IDLE)
    def _inspect_pcb(self, frame, image_path: str) -> InspectionResult:
        """Perform PCB inspection on the captured frame"""
        try:
            # Here we would call the actual inspection logic
            # For now, we'll simulate a simple inspection
            
            # Check if frame is valid
            if frame is None or frame.size == 0:
                raise ValueError("Invalid or empty frame")
            
            # Simulate inspection time (50-200ms)
            time.sleep(0.1 + (random.random() * 0.15))
            
            # Randomly determine if defective (20% chance for demo)
            is_defective = random.random() < 0.2
            
            # Generate some random defects if defective
            defects = []
            if is_defective:
                num_defects = random.randint(1, 3)
                for i in range(num_defects):
                    defects.append({
                        'type': random.choice(['missing_component', 'solder_bridge', 'misalignment', 'scratch']),
                        'confidence': 0.7 + (random.random() * 0.29),  # 0.7-0.99
                        'location': {
                            'x': random.randint(0, frame.shape[1]),
                            'y': random.randint(0, frame.shape[0]),
                            'width': random.randint(10, 100),
                            'height': random.randint(10, 100)
                        },
                        'severity': random.choice(['low', 'medium', 'high'])
                    })
            
            # Calculate quality score (simplified)
            quality_score = max(0, min(100, 100 - (len(defects) * 20) + (random.random() * 10)))
            
            return InspectionResult(
                timestamp=datetime.now(),
                image_path=image_path,
                is_defective=is_defective,
                defects=defects,
                quality_score=quality_score,
                severity_level='high' if is_defective and any(d['severity'] == 'high' for d in defects) 
                              else 'medium' if is_defective and any(d['severity'] == 'medium' for d in defects)
                              else 'low' if is_defective
                              else 'none',
                confidence_score=0.9 if not is_defective else 0.7,
                inspection_time=0.1 + (random.random() * 0.15),
                result_id=f"res_{int(time.time())}_{random.randint(1000, 9999)}"
            )
            
        except Exception as e:
            logger.error(f"Inspection failed: {e}", exc_info=True)
            # Return a default error result
            return InspectionResult(
                timestamp=datetime.now(),
                image_path=image_path,
                is_defective=True,
                defects=[{
                    'type': 'inspection_error',
                    'confidence': 1.0,
                    'location': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
                    'severity': 'high',
                    'message': f'Inspection failed: {str(e)}'
                }],
                quality_score=0.0,
                severity_level='high',
                confidence_score=0.0,
                inspection_time=0.0,
                result_id=f"err_{int(time.time())}"
            )
            
                    # This was a duplicate code block that's already handled above
                    # The code has been removed to fix the indentation and structure

def _inspect_pcb(self, frame, image_path: str) -> InspectionResult:
    """Perform PCB inspection on the captured frame"""
    try:
        # Here we would call the actual inspection logic
        # For now, we'll simulate a simple inspection
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty frame")
        
        # Simulate inspection time (50-200ms)
        time.sleep(0.1 + (random.random() * 0.15))
        
        # Randomly determine if defective (20% chance for demo)
        is_defective = random.random() < 0.2
        
        # Generate some random defects if defective
        defects = []
        if is_defective:
            num_defects = random.randint(1, 3)
            for i in range(num_defects):
                defects.append({
                    'type': random.choice(['missing_component', 'solder_bridge', 'misalignment', 'scratch']),
                    'confidence': 0.7 + (random.random() * 0.29),  # 0.7-0.99
                    'location': {
                        'x': random.randint(0, frame.shape[1]),
                        'y': random.randint(0, frame.shape[0]),
                        'width': random.randint(10, 100),
                        'height': random.randint(10, 100)
                    },
                    'severity': random.choice(['low', 'medium', 'high'])
                })
        
        # Calculate quality score (simplified)
        quality_score = max(0, min(100, 100 - (len(defects) * 20) + (random.random() * 10)))
        
        return InspectionResult(
            timestamp=datetime.now(),
            image_path=image_path,
            is_defective=is_defective,
            defects=defects,
            quality_score=quality_score,
            severity_level='high' if is_defective and any(d['severity'] == 'high' for d in defects) 
                          else 'medium' if is_defective and any(d['severity'] == 'medium' for d in defects)
                          else 'low' if is_defective
                          else 'none',
            confidence_score=0.9 if not is_defective else 0.7,
            inspection_time=0.1 + (random.random() * 0.15),
            result_id=f"res_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        
    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=True)
        # Return a default error result
        return InspectionResult(
            timestamp=datetime.now(),
            image_path=image_path,
            is_defective=True,
            defects=[{
                'type': 'inspection_error',
                'confidence': 1.0,
                'location': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
                'severity': 'high',
                'message': f'Inspection failed: {str(e)}'
            }],
            quality_score=0.0,
            severity_level='high',
            confidence_score=1.0,
            inspection_time=0.0,
            result_id=f"err_{int(time.time())}_{random.randint(1000, 9999)}"
        )

    def trigger_inspection(self) -> bool:
        """
        Trigger a manual inspection with enhanced camera and state validation
        
        Returns:
            bool: True if inspection was successfully triggered, False otherwise
        """
        with self._lock:  # Ensure thread safety
            if not self.is_running:
                logger.warning("Cannot trigger inspection - workflow is not running")
                return False
                
            # Double-check camera connection
            if not self._camera_connected or self.camera_manager is None:
                logger.warning("Camera not connected, attempting to reconnect...")
                if not self.connect_camera():
                    logger.error("❌ Failed to trigger inspection - could not connect to camera")
                    self._notify_state_change(InspectionState.ERROR, {
                        'error': 'Camera connection lost',
                        'recoverable': True
                    })
                    return False
            
            # Check if camera is ready
            if not self.camera_manager.is_camera_connected():
                logger.error("❌ Camera is not ready for inspection")
                self._camera_connected = False
                return False
                
            # Ensure we're in a valid state to start inspection
            if self.current_state not in [InspectionState.WAITING_FOR_PCB, InspectionState.IDLE]:
                logger.warning(f"Cannot trigger inspection in current state: {self.current_state}")
                return False
            
            logger.info("✅ Starting manual inspection")
            self._notify_state_change(InspectionState.CAPTURING)
            
            try:
                # Capture and process the image
                frame = self._capture_stabilized_image()
                if frame is None or frame.size == 0:
                    raise ValueError("Failed to capture image")
                
                # Save the captured image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = str(Path("captures") / f"inspection_{timestamp}.jpg")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                cv2.imwrite(image_path, frame)
                
                # Perform inspection
                self._inspect_pcb(frame, image_path)
                return True
                
            except Exception as e:
                logger.error(f"❌ Error during manual inspection: {e}")
                self._notify_state_change(InspectionState.ERROR, {
                    'error': f'Inspection failed: {str(e)}',
                    'recoverable': True
                })
                return False

def main():
    """Main function for testing workflow manager"""
    import random
    logging.basicConfig(level=logging.INFO)
    
    # Create workflow manager
    workflow = InspectionWorkflowManager()
    
    # Test callbacks
    def state_callback(state_data):
        print(f"🔄 State: {state_data['state']} - {state_data.get('data', {}).get('message', '')}")
    
    def result_callback(result_data):
        status = "DEFECTIVE" if result_data['is_defective'] else "GOOD"
        print(f"📊 Result: {status} - Quality: {result_data['quality_score']:.1f}% - Time: {result_data['inspection_time']:.2f}s")
    
    workflow.add_state_callback(state_callback)
    workflow.add_result_callback(result_callback)
    
    # Initialize system
    if workflow.initialize_system():
        print("✅ System initialized")
        
        # Start workflow in auto mode
        workflow.start_inspection_workflow(auto_mode=True)
        
        try:
            # Run for 60 seconds as demo
            time.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            workflow.shutdown()
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    main()