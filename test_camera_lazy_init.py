"""
Test script for verifying lazy camera initialization in the PCB inspection system.
"""
import time
import logging
from realtime_workflow import InspectionWorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_camera_lazy_initialization():
    """Test that camera is only initialized when explicitly connected."""
    logger.info("=== Starting Camera Lazy Initialization Test ===")
    
    # Create workflow manager
    logger.info("Creating InspectionWorkflowManager...")
    workflow = InspectionWorkflowManager()
    
    # Verify camera is not connected initially
    logger.info("1. Checking initial camera state...")
    assert not workflow.is_camera_connected(), "Camera should not be connected initially"
    logger.info("✓ Camera is not connected initially")
    
    # Initialize system (should not connect to camera)
    logger.info("2. Initializing system...")
    assert workflow.initialize_system(), "System initialization failed"
    logger.info("✓ System initialized successfully")
    
    # Verify camera is still not connected
    logger.info("3. Verifying camera is still not connected...")
    assert not workflow.is_camera_connected(), "Camera should still not be connected after system init"
    logger.info("✓ Camera is still not connected after system init")
    
    # Manually connect to camera
    logger.info("4. Manually connecting to camera...")
    assert workflow.connect_camera(), "Failed to connect to camera"
    logger.info("✓ Camera connected successfully")
    
    # Verify camera is now connected
    logger.info("5. Verifying camera is now connected...")
    assert workflow.is_camera_connected(), "Camera should now be connected"
    logger.info("✓ Camera is now connected")
    
    # Test capturing an image
    logger.info("6. Testing image capture...")
    image = workflow._capture_stabilized_image()
    assert image is not None, "Failed to capture image"
    logger.info(f"✓ Successfully captured image: {image.shape}")
    
    # Disconnect camera
    logger.info("7. Disconnecting camera...")
    assert workflow.disconnect_camera(), "Failed to disconnect camera"
    logger.info("✓ Camera disconnected successfully")
    
    # Verify camera is disconnected
    logger.info("8. Verifying camera is disconnected...")
    assert not workflow.is_camera_connected(), "Camera should be disconnected"
    logger.info("✓ Camera is now disconnected")
    
    # Clean up
    workflow.shutdown()
    logger.info("=== Test completed successfully ===\n")

if __name__ == "__main__":
    test_camera_lazy_initialization()
