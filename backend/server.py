import os
import sys
import logging
import time
from pathlib import Path

# Add the parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.resolve())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request, Form
from realtime_workflow import InspectionState
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tempfile
import shutil
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import cv2
import numpy as np
import json
import csv
import base64
from io import BytesIO
from PIL import Image
import asyncio
import threading
import gc

# Configure logging with UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg.replace('✅', '[OK]') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure root logger – skip FileHandler on Render (ephemeral filesystem)
_log_handlers = [UTF8StreamHandler()]
try:
    _log_handlers.append(logging.FileHandler('app.log', encoding='utf-8'))
except (OSError, PermissionError):
    pass  # on some cloud hosts writing files here may fail

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers
)
logger = logging.getLogger(__name__)

# Import our enhanced PCB modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from enhanced_pcb_inspection import IndustrialPCBInspector
from camera_integration import CameraManager
from realtime_workflow import InspectionWorkflowManager


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection – use .get() so missing env vars produce a clear error
# instead of a raw KeyError that crashes the process silently.
mongo_url = os.environ.get('MONGO_URL')
db_name = os.environ.get('DB_NAME', 'pcb_inspection')

if not mongo_url:
    logger.error(
        "MONGO_URL environment variable is NOT set! "
        "Set it in the Render dashboard → Environment tab. "
        "The server will start but database operations will fail."
    )
    # Use a dummy client so the server can at least start and serve /health
    client = None
    db = None
else:
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]

# Create the main app with CORS and settings
app = FastAPI(
    title="PCB Inspection API",
    description="API for Automated PCB Inspection System",
    version="1.0.0"
)

# ---------- CORS: belt-and-suspenders approach ----------
# 1) Standard Starlette CORSMiddleware (handles most cases)
ALLOWED_ORIGINS = [
    "https://pcb-main.onrender.com",
    "https://pcb-main-frontend.onrender.com",
    "https://pcb-inspection-frontend.onrender.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 2) Raw ASGI middleware that ALWAYS injects CORS headers.
#    BaseHTTPMiddleware is NOT used because it has known bugs with file-upload
#    (multipart) requests that can silently drop headers or break the body stream.
from starlette.types import ASGIApp, Receive, Scope, Send

class ForceCORSHeadersMiddleware:
    """Pure-ASGI CORS middleware – zero interference with request body."""

    CORS_HEADERS = [
        (b"access-control-allow-origin", b"*"),
        (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS, PATCH"),
        (b"access-control-allow-headers", b"*"),
        (b"access-control-expose-headers", b"*"),
    ]

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast-path: handle preflight OPTIONS
        if scope["method"] == "OPTIONS":
            headers = self.CORS_HEADERS + [
                (b"access-control-max-age", b"86400"),
            ]
            await send({"type": "http.response.start", "status": 200, "headers": headers})
            await send({"type": "http.response.body", "body": b""})
            return

        # For all other requests, intercept the response start to inject headers
        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Only add if not already present
                existing = {h[0].lower() for h in headers}
                for key, value in self.CORS_HEADERS:
                    if key not in existing:
                        headers.append((key, value))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_cors)
        except Exception:
            # Even on unhandled crash, return a 500 WITH CORS headers
            headers = self.CORS_HEADERS + [(b"content-type", b"text/plain")]
            await send({"type": "http.response.start", "status": 500, "headers": headers})
            await send({"type": "http.response.body", "body": b"Internal Server Error"})

app.add_middleware(ForceCORSHeadersMiddleware)
# ---------- END CORS ----------

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Create required directories
for directory in ["defective_storage", "dataset/good", "dataset/defective"]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Note: Static files mount removed to prevent CORS conflicts
# All file serving is now handled through the API endpoint below

@api_router.get("/pcb/defective/{filename}")
async def get_defective_image(filename: str):
    """Serve defective PCB images"""
    try:
        return FileResponse(
            os.path.join(ROOT_DIR, "..", "defective_storage", filename),
            media_type="image/jpeg"
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class PCBInspectionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_name: str
    is_defective: bool
    saved_filename: Optional[str] = None
    confidence_score: float = 0.0
    defects: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_shape: Optional[List[int]] = None
    visualization_path: Optional[str] = None

class PCBInspectionCreate(BaseModel):
    image_name: str
    
# Initialize Enhanced PCB Inspector
PROJECT_ROOT = ROOT_DIR.parent
pcb_inspector = IndustrialPCBInspector(
    dataset_path=str(PROJECT_ROOT / "dataset"),
    test_path=str(PROJECT_ROOT / "test_images"),
    defective_storage=str(PROJECT_ROOT / "defective_storage"),
)

# Initialize directory paths for file operations
BASE_DIR = Path(__file__).resolve().parent
ANNOTATED_DIR = BASE_DIR.parent / "defective_storage"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG_PATH = BASE_DIR / "inspections.csv"
REALTIME_LAST = ANNOTATED_DIR / "realtime" / "last_annotated.jpg"

# Camera and Workflow initialization is deferred to the startup event below
MAX_RETRIES = 3

def log_camera_info(cameras):
    """Log information about available cameras"""
    if not cameras:
        logger.warning("⚠️ No cameras found. Please check camera connections.")
        return
        
    logger.info(f"✅ Found {len(cameras)} available cameras")
    for i, cam in enumerate(cameras):
        cam_name = cam.get('name', f'Camera {i}')
        cam_type = cam.get('type', 'Unknown')
        resolution = cam.get('resolution', 'Unknown')
        fps = cam.get('fps', 'N/A')
        status = cam.get('status', 'Unknown')
        
        logger.info(f"  {i+1}. {cam_name} ({cam_type})")
        logger.info(f"     Resolution: {resolution}, FPS: {fps}, Status: {status}")

# Function to initialize camera with retries
def initialize_camera():
    """Initialize camera manager with retry logic"""
    from camera_integration import CameraManager
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"🔍 Attempting to initialize camera manager (attempt {attempt + 1}/{MAX_RETRIES})...")
            cm = CameraManager()
            
            # Initialize the camera manager
            if not cm.initialize():
                raise RuntimeError("Failed to initialize camera manager")
            
            # Discover available cameras
            available_cameras = cm.discover_cameras()
            log_camera_info(available_cameras)
            
            if available_cameras:
                return cm
                
        except ImportError as e:
            logger.error(f"❌ Required dependencies not found: {e}")
            if attempt == MAX_RETRIES - 1:
                logger.error("Please install the required dependencies and try again.")
            time.sleep(1)
            continue
            
        except Exception as e:
            logger.error(f"⚠️ Camera initialization attempt {attempt + 1} failed: {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                logger.error("❌ Failed to initialize camera after multiple attempts")
                return None
            time.sleep(1)  # Wait before retry
    
    return None

# --- Lazy initialization: camera + workflow are set up at startup event, not at
# module load.  This avoids blocking import on headless servers (Render) where
# cv2.VideoCapture probes 10 indices with Windows-only backends for ~30 s.
camera_manager = None
workflow_manager = None

@app.on_event("startup")
async def _lazy_init_camera_and_workflow():
    """Initialize camera & workflow AFTER the server is listening."""
    global camera_manager, workflow_manager
    import platform
    # Skip camera probing entirely on Linux (Render is headless, no webcam)
    if platform.system() != "Windows":
        logger.info("Headless Linux detected – skipping camera initialization")
        camera_manager = None
    else:
        camera_manager = initialize_camera()

    try:
        workflow_manager = InspectionWorkflowManager(pcb_inspector=pcb_inspector)
        if camera_manager:
            workflow_manager.camera_manager = camera_manager
            logger.info("Workflow manager initialized with camera support")
        else:
            logger.warning("Workflow manager initialized without camera support")
    except Exception as e:
        logger.error(f"Failed to initialize workflow manager: {e}")
        workflow_manager = None

    # Register workflow callbacks
    if workflow_manager is not None:
        workflow_manager.add_state_callback(on_state_change)
        workflow_manager.add_result_callback(on_inspection_result)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

websocket_manager = ConnectionManager()

# Global variables to track reference model status
reference_model_built = False
training_in_progress = False
training_error = None

async def ensure_reference_model():
    """Ensure reference model is built from good PCB images"""
    global reference_model_built
    if not reference_model_built:
        logger.info("Building PCB reference model...")
        success = pcb_inspector.build_industrial_reference_model()
        reference_model_built = success
        if success:
            logger.info("✅ PCB reference model built successfully")
        else:
            logger.warning("⚠️ Could not build reference model - no good PCB images found")
    return reference_model_built

# Workflow callbacks
def on_state_change(state_data):
    """Handle workflow state changes"""
    asyncio.create_task(websocket_manager.broadcast({
        'type': 'state_change',
        'data': state_data
    }))

def on_inspection_result(result_data):
    """Handle inspection results"""
    asyncio.create_task(websocket_manager.broadcast({
        'type': 'inspection_result', 
        'data': result_data
    }))

# NOTE: workflow callbacks are registered inside _lazy_init_camera_and_workflow()

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "PCB Automated Optical Inspection System API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    if db is not None:
        _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    if db is None:
        return []
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

def _train_model_background():
    """Background task to train the model without blocking the API"""
    global reference_model_built, training_in_progress, training_error
    try:
        logger.info("Starting model training in background...")
        gc.collect()  # Free memory before training
        success = pcb_inspector.build_industrial_reference_model()
        reference_model_built = success
        training_in_progress = False
        
        if success:
            logger.info("Model training completed successfully")
            training_error = None
        else:
            logger.warning("Model training failed - no good PCB images")
            training_error = "No good PCB images found"
    except MemoryError:
        logger.error("Out of memory during training! Dataset too large for server RAM.")
        training_error = "Out of memory - server has insufficient RAM for training"
        training_in_progress = False
        gc.collect()
    except Exception as e:
        logger.error(f"Error training PCB model: {e}")
        training_error = str(e)
        training_in_progress = False
        gc.collect()

@api_router.post("/pcb/train")
async def train_pcb_model():
    """Start reference model training in background"""
    global training_in_progress, training_error
    
    # Check if already training
    if training_in_progress:
        return {
            "message": "Training already in progress",
            "success": False,
            "training_in_progress": True
        }
    
    # Check if already trained
    if reference_model_built:
        return {
            "message": "Model already trained",
            "success": True,
            "training_in_progress": False,
            "already_trained": True
        }
    
    # Start training in background thread
    training_in_progress = True
    training_error = None
    thread = threading.Thread(target=_train_model_background, daemon=True)
    thread.start()
    
    return {
        "message": "Model training started in background. This may take 15-30 seconds.",
        "success": True,
        "training_in_progress": True
    }

@api_router.get("/pcb/train/status")
async def get_train_status():
    """Get current training status"""
    return {
        "training_in_progress": training_in_progress,
        "model_trained": reference_model_built,
        "error": training_error
    }

@api_router.post("/pcb/inspect")
async def inspect_pcb_image(
    file: UploadFile = File(...)
):
    """
    Inspect uploaded PCB image for defects
    
    Args:
        file: The image file to inspect
    """
    try:
        
        # Read the uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del contents, nparr  # free upload buffer immediately
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # --- Resize to prevent OOM on Render (max 1024px on longest side) ---
        MAX_DIM = 1024
        h, w = image.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            logger.info(f"Resized upload from {w}x{h} to {image.shape[1]}x{image.shape[0]} for inspection")
        
        # Perform inspection
        results = pcb_inspector.detect_industrial_defects(image, source_mode="manual")
        
        # Create industrial visualization
        annotated_image = pcb_inspector.create_industrial_visualization(image, results)
        
        # Drop heavy intermediate data that is no longer needed
        results.pop('processed_images', None)
        results.pop('features', None)
        
        # Save annotated image temporarily
        temp_filename = f"temp_inspection_{uuid.uuid4().hex}.jpg"
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = str(ANNOTATED_DIR / temp_filename)
        cv2.imwrite(temp_path, annotated_image)
        
        # Save defective image and prepare visualization path
        saved_filename = None
        if results["is_defective"]:
            # Generate a unique filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"defective_pcb_{timestamp}.jpg"
            save_path = os.path.join(ROOT_DIR, "..", "defective_storage", saved_filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the annotated image
            cv2.imwrite(save_path, annotated_image)
            
            # Set the visualization path to the saved image
            visualization_path = f"/api/pcb/defective/{saved_filename}"
        else:
            # For non-defective images, use the temporary result
            visualization_path = f"/api/pcb/result/{temp_filename}"
        
        # Handle potential None filename with fallback
        safe_filename = file.filename or f"uploaded_image_{uuid.uuid4().hex[:8]}.jpg"
        
        # Save the result with visualization path
        result_data = PCBInspectionResult(
            image_name=safe_filename,
            is_defective=results['is_defective'],
            confidence_score=float(results['confidence_score']) if results['confidence_score'] is not None else 0.0,
            defects=results['defects'],
            image_shape=list(image.shape) if hasattr(image, 'shape') else None,
            visualization_path=visualization_path,
            saved_filename=saved_filename  # Also save the filename for reference
        )
        
        # Store in database
        try:
            if db is not None:
                await db.pcb_inspections.insert_one(result_data.dict())
        except Exception as db_err:
            logger.warning(f"DB insert failed: {db_err}")
        
        # Convert to JSON-serializable format
        result_dict = result_data.dict()
        result_dict['id'] = str(result_dict['id'])
        result_dict['timestamp'] = result_dict['timestamp'].isoformat()
        
        # Save to CSV file for tracking and download
        csv_record = {
            "id": result_dict['id'],
            "timestamp": result_dict['timestamp'],
            "mode": "api",
            "filename": safe_filename,
            "is_defective": str(results['is_defective']).lower(),
            "confidence_score": float(results['confidence_score']) if results['confidence_score'] is not None else 0.0,
            "defect_count": len(results['defects']),
            "defects": results['defects'],
            "annotated_relpath": visualization_path
        }
        _append_csv(csv_record)
        
        # Free heavy objects before returning
        del image, annotated_image
        gc.collect()
        
        return result_dict
        
    except Exception as e:
        gc.collect()
        logger.error(f"Error inspecting PCB: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/pcb/result/{filename}")
async def get_inspection_result_image(filename: str):
    """Get inspection result visualization image"""
    file_path = str(ANNOTATED_DIR / filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result image not found")
    
    return FileResponse(file_path, media_type="image/jpeg")

@api_router.get("/pcb/inspections")
async def get_pcb_inspections():
    """Get all PCB inspection results from CSV file"""
    try:
        # Ensure CSV file exists with headers
        try:
            if not CSV_LOG_PATH.exists() or os.path.getsize(CSV_LOG_PATH) == 0:
                CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(CSV_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "id", "timestamp", "mode", "filename", 
                        "is_defective", "confidence_score", 
                        "defect_count", "defects", "annotated_relpath"
                    ])
        except Exception as e:
            logger.error(f"Error ensuring CSV headers: {e}")
            return []
            
        rows = []
        
        # Check if file exists and has content
        if not CSV_LOG_PATH.exists() or os.path.getsize(CSV_LOG_PATH) == 0:
            return []
            
        with open(CSV_LOG_PATH, 'r', encoding='utf-8') as f:
            # Read all lines and filter out empty ones
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # If only header exists or file is empty
            if len(lines) <= 1:
                return []
                
            # Parse CSV data
            reader = csv.DictReader(lines)
            for row in reader:
                try:
                    # Skip empty rows or rows with missing required fields
                    if not row.get('id') or not row.get('filename'):
                        continue
                        
                    # Convert types with proper error handling
                    try:
                        row["is_defective"] = str(row.get("is_defective", "")).lower() == "true"
                        row["confidence_score"] = float(row.get("confidence_score", 0) or 0)
                        row["defect_count"] = int(row.get("defect_count", 0) or 0)
                        
                        # Parse defects field
                        defects = row.get("defects", "0")
                        try:
                            if defects.isdigit():
                                count = int(defects)
                                row["defects"] = [{"type": "unknown"} for _ in range(count)]
                            elif defects.startswith('[') and defects.endswith(']'):
                                try:
                                    row["defects"] = json.loads(defects)
                                except:
                                    row["defects"] = []
                            else:
                                row["defects"] = []
                        except Exception:
                            row["defects"] = []
                        
                        # Add visualization path
                        annotated_path = row.get('annotated_relpath', '').strip()
                        if annotated_path:
                            row["visualization_path"] = annotated_path
                        else:
                            row["visualization_path"] = None
                        
                        rows.append(row)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid row {row}: {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error parsing row: {str(e)}")
                    continue
        
        # Sort by timestamp descending (newest first)
        rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        logger.info(f"Returning {len(rows)} inspections from CSV, newest first")
        
        return rows
        
    except Exception as e:
        logger.error(f"Error getting inspections from CSV: {str(e)}")
        return []

@api_router.get("/pcb/stats")
async def get_pcb_stats():
    """Get PCB inspection statistics from CSV"""
    try:
        # Read from CSV file instead of MongoDB
        if not CSV_LOG_PATH.exists() or os.path.getsize(CSV_LOG_PATH) == 0:
            return {
                "total_inspections": 0,
                "defective_count": 0,
                "good_count": 0,
                "defect_rate": 0,
                "defect_types": [],
                "reference_model_available": reference_model_built,
                "training_in_progress": training_in_progress,
                "session_stats": {},
                "realtime_available": True
            }
        
        total_inspections = 0
        defective_count = 0
        defect_type_counts = {}
        
        with open(CSV_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) <= 1:
                return {
                    "total_inspections": 0,
                    "defective_count": 0,
                    "good_count": 0,
                    "defect_rate": 0,
                    "defect_types": [],
                    "reference_model_available": reference_model_built,
                    "training_in_progress": training_in_progress,
                    "session_stats": {},
                    "realtime_available": True
                }
            
            reader = csv.DictReader(lines)
            for row in reader:
                try:
                    if not row.get('id') or not row.get('filename'):
                        continue
                    
                    total_inspections += 1
                    
                    # Count defective
                    is_defective = str(row.get("is_defective", "")).lower() == "true"
                    if is_defective:
                        defective_count += 1
                    
                    # Count defect types
                    defects_str = row.get("defects", "")
                    if defects_str.startswith('[') and defects_str.endswith(']'):
                        try:
                            defects = json.loads(defects_str)
                            for defect in defects:
                                defect_type = defect.get("type", "unknown")
                                defect_type_counts[defect_type] = defect_type_counts.get(defect_type, 0) + 1
                        except:
                            pass
                
                except Exception as e:
                    logger.warning(f"Skipping row in stats calculation: {str(e)}")
                    continue
        
        good_count = total_inspections - defective_count
        
        # Format defect types
        defect_types = [{"type": k, "count": v} for k, v in defect_type_counts.items()]
        defect_types.sort(key=lambda x: x["count"], reverse=True)
        
        # Get workflow session stats if available
        session_stats = {}
        if workflow_manager:
            session_stats = workflow_manager.get_session_stats()
        
        return {
            "total_inspections": total_inspections,
            "defective_count": defective_count,
            "good_count": good_count,
            "defect_rate": defective_count / total_inspections if total_inspections > 0 else 0,
            "defect_types": defect_types,
            "reference_model_available": reference_model_built,
            "training_in_progress": training_in_progress,
            "session_stats": session_stats,
            "realtime_available": True
        }
    except Exception as e:
        logger.error(f"Error getting PCB stats: {e}")
        # Return empty stats instead of raising exception to avoid CORS issues
        return {
            "total_inspections": 0,
            "defective_count": 0,
            "good_count": 0,
            "defect_rate": 0,
            "defect_types": [],
            "reference_model_available": reference_model_built,
            "training_in_progress": training_in_progress,
            "session_stats": {},
            "realtime_available": True,
            "error": str(e)
        }

# Real-time inspection endpoints
@api_router.get("/pcb/realtime/available-cameras")
async def get_available_cameras():
    """
    Get available cameras with detailed status information
    
    Returns:
        dict: Dictionary containing list of available cameras and their status
    """
    try:
        if not camera_manager:
            logger.error("Camera manager not initialized when discovering cameras")
            return {
                "status": "error",
                "message": "Camera manager not initialized",
                "cameras": [],
                "count": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        try:
            cameras = camera_manager.discover_cameras()
            logger.info(f"Discovered {len(cameras)} available cameras")
            
            return {
                "status": "success",
                "cameras": cameras,
                "count": len(cameras),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error discovering cameras: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to discover cameras: {str(e)}",
                "cameras": [],
                "count": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.critical(f"Unexpected error in get_available_cameras: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "An unexpected error occurred while discovering cameras",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@api_router.post("/realtime/camera/connect")
async def connect_camera(camera_type: str = "opencv"):
    """
    Connect to the specified camera type with retry logic and detailed status reporting
    
    Args:
        camera_type: Type of camera to connect ('opencv' or 'baumer')
        
    Returns:
        dict: Connection status and camera information
    """
    global camera_manager
    max_retries = 3
    retry_delay = 1.0  # seconds
    
    logger.info(f"🔌 Attempting to connect to {camera_type} camera...")
    
    try:
        # Initialize camera manager if not already done
        if camera_manager is None:
            camera_manager = initialize_camera()
            if camera_manager is None:
                error_msg = "Failed to initialize camera manager"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Connect to the specified camera type with retries
        for attempt in range(max_retries):
            try:
                if camera_type.lower() == "baumer":
                    if not camera_manager.connect_camera(camera_type="baumer"):
                        error_msg = "Failed to connect to Baumer camera. Please ensure it is properly connected and powered on."
                        logger.error(error_msg)
                        return {
                            "status": "error",
                            "message": error_msg,
                            "requires_baumer": True,
                            "suggestion": "Please connect a Baumer camera and ensure the drivers are installed.",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                else:  # Default to OpenCV
                    if not camera_manager.connect_camera(camera_type="opencv"):
                        error_msg = "Failed to connect to webcam. Please check if it's connected and not in use by another application."
                        logger.error(error_msg)
                        return {
                            "status": "error",
                            "message": error_msg,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                
                # If we get here, connection was successful
                break
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    error_msg = f"Failed to connect to {camera_type} camera: {str(e)}"
                    logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                    
                    # Special handling for Baumer camera errors
                    if camera_type.lower() == "baumer":
                        error_details = (
                            "Please ensure:\n"
                            "1. The Baumer camera is properly connected\n"
                            "2. The Baumer GenICam drivers are installed\n"
                            "3. No other application is using the camera"
                        )
                    else:
                        error_details = str(e)
                    
                    return {
                        "status": "error",
                        "message": error_msg,
                        "details": error_details,
                        "requires_baumer": (camera_type.lower() == "baumer"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Wait before retrying
                await asyncio.sleep(retry_delay)
        
        # Get camera info
        camera_info = camera_manager.get_camera_info()
        
        logger.info(f"✅ Successfully connected to {camera_type} camera")
        return {
            "status": "success",
            "message": f"Successfully connected to {camera_type} camera",
            "camera_type": camera_type,
            "camera_info": camera_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Unexpected error connecting to {camera_type} camera: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "status": "error",
            "message": error_msg,
            "requires_baumer": (camera_type.lower() == "baumer"),
            "timestamp": datetime.utcnow().isoformat()
        }

@api_router.post("/realtime/camera/disconnect")
async def disconnect_camera():
    """Disconnect camera"""
    try:
        if camera_manager is None:
            logger.warning("Camera manager not initialized")
            return {
                "success": False, 
                "message": "Camera manager not initialized",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        camera_manager.disconnect_camera()
        return {
            "success": True, 
            "message": "Camera disconnected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error disconnecting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/realtime/workflow/start")
async def start_workflow(auto_mode: bool = False):
    """
    Start the real-time inspection workflow with comprehensive status reporting
    
    Args:
        auto_mode: Whether to run in automatic inspection mode
        
    Returns:
        dict: Workflow status and initialization details
    """
    if not workflow_manager:
        error_msg = "Workflow manager not initialized"
        logger.error(error_msg)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    if workflow_manager.is_running:
        status_msg = "Workflow is already running"
        logger.warning(status_msg)
        return {
            "status": "already_running",
            "message": status_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Verify camera is connected before starting workflow
        if not camera_manager or not camera_manager.is_camera_connected():
            error_msg = "Cannot start workflow: No camera connected"
            logger.error(error_msg)
            
            # Attempt to connect to default camera
            try:
                await connect_camera("opencv")
            except Exception as e:
                logger.error(f"Failed to automatically connect camera: {e}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "status": "error",
                        "message": error_msg,
                        "suggestion": "Please connect a camera first using /realtime/camera/connect",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
        
        # Start the workflow
        logger.info(f"Starting workflow in {'auto' if auto_mode else 'manual'} mode")
        workflow_manager.start_inspection_workflow(auto_mode=auto_mode)
        
        # Notify connected clients
        await websocket_manager.broadcast({
            "type": "workflow_status",
            "status": "started",
            "auto_mode": auto_mode,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "message": f"Workflow started in {'auto' if auto_mode else 'manual'} mode",
            "auto_mode": auto_mode,
            "camera_connected": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
        
    except Exception as e:
        error_msg = f"Failed to start workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Notify connected clients
        await websocket_manager.broadcast({
            "type": "workflow_status",
            "status": "error",
            "message": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@api_router.post("/realtime/workflow/trigger")
async def trigger_manual_inspection():
    """
    Trigger a manual inspection with comprehensive status reporting
    
    Returns:
        dict: Status of the manual inspection trigger
    """
    if not workflow_manager:
        error_msg = "Workflow manager not initialized"
        logger.error(error_msg)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    if not workflow_manager.is_running:
        error_msg = "Cannot trigger inspection - workflow is not running"
        logger.warning(error_msg)
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": error_msg,
                "suggestion": "Start the workflow first using /realtime/workflow/start",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    try:
        logger.info("Triggering manual inspection...")
        if not workflow_manager:
            error_msg = "Workflow manager not initialized"
            logger.error(error_msg)
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "message": error_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        success = workflow_manager.trigger_inspection()
        
        if success:
            status_msg = "Manual inspection triggered successfully"
            logger.info(status_msg)
            
            # Notify connected clients
            await websocket_manager.broadcast({
                "type": "inspection_triggered",
                "status": "success",
                "message": status_msg,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "status": "success",
                "message": status_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            error_msg = "Failed to trigger manual inspection - workflow not ready"
            logger.warning(error_msg)
            
            # Notify connected clients
            await websocket_manager.broadcast({
                "type": "inspection_triggered",
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": error_msg,
                    "suggestion": "Wait for the workflow to be ready before triggering",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        error_msg = f"Error triggering manual inspection: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Notify connected clients
        await websocket_manager.broadcast({
            "type": "inspection_triggered",
            "status": "error",
            "message": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@api_router.get("/realtime/workflow/state")
async def get_workflow_state():
    """
    Get the current workflow state with detailed status information
    
    Returns:
        dict: Current workflow state and related information
    """
    if not workflow_manager:
        error_msg = "Workflow manager not initialized"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "state": "unavailable",
            "camera_connected": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get the current state with additional context
        if not workflow_manager:
            error_msg = "Workflow manager not initialized"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "state": "unavailable",
                "camera_connected": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        state = workflow_manager.get_current_state()
        
        # Add camera connection status
        camera_status = {
            "camera_connected": camera_manager.is_camera_connected() if camera_manager else False,
            "camera_info": camera_manager.get_camera_info() if camera_manager else None
        }
        
        # Add system status
        system_status = {
            "cpu_usage": 0,  # TODO: Implement system monitoring
            "memory_usage": 0,  # TODO: Implement system monitoring
            "disk_space": 0,  # TODO: Implement system monitoring
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Combine all status information
        response = {
            "status": "success",
            "state": state,
            "workflow_running": workflow_manager.is_running,
            "auto_mode": workflow_manager.auto_mode if hasattr(workflow_manager, 'auto_mode') else False,
            "camera": camera_status,
            "system": system_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        error_msg = f"Error getting workflow state: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "status": "error",
            "message": error_msg,
            "state": "error",
            "camera_connected": False,
            "timestamp": datetime.utcnow().isoformat()
        }

@api_router.get("/realtime/workflow/stats")
async def get_workflow_stats():
    """
    Get workflow statistics and recent results
    
    Returns:
        dict: Workflow statistics including session stats and recent results
    """
    try:
        if not workflow_manager:
            logger.warning("Workflow manager not initialized")
            return {
                "session_stats": {
                    "total_inspected": 0,
                    "total_defective": 0,
                    "total_good": 0,
                    "session_duration": 0,
                    "avg_inspection_time": 0,
                    "avg_quality_score": 100.0
                },
                "recent_results": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get session statistics
        session_stats = workflow_manager.get_session_stats() if workflow_manager else {
            "total_inspected": 0,
            "total_defective": 0,
            "total_good": 0,
            "defect_rate": 0,
            "avg_quality": 100.0
        }
        
        # Get recent results (last 20)
        recent_results = []
        if hasattr(workflow_manager, 'results_history'):
            recent_results = [
                {
                    "result_id": result.result_id,
                    "timestamp": result.timestamp.isoformat(),
                    "is_defective": result.is_defective,
                    "quality_score": result.quality_score,
                    "defect_count": len(result.defects),
                    "inspection_time": result.inspection_time
                }
                for result in workflow_manager.results_history[-20:]  # Last 20 results
            ]
        
        return {
            "session_stats": session_stats,
            "recent_results": recent_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow stats: {e}")
        return {
            "session_stats": {
                "total_inspected": 0,
                "total_defective": 0,
                "total_good": 0,
                "session_duration": 0,
                "avg_inspection_time": 0,
                "avg_quality_score": 100.0
            },
            "recent_results": [],
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@api_router.post("/realtime/workflow/stop")
async def stop_workflow():
    """
    Stop the real-time inspection workflow with proper cleanup
    
    Returns:
        dict: Status of the workflow stop operation
    """
    if not workflow_manager:
        error_msg = "Workflow manager not initialized"
        logger.error(error_msg)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    if not workflow_manager.is_running:
        status_msg = "Workflow is not running"
        logger.warning(status_msg)
        return {
            "status": "not_running",
            "message": status_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        logger.info("Stopping workflow...")
        
        # Notify clients that we're stopping
        await websocket_manager.broadcast({
            "type": "workflow_status",
            "status": "stopping",
            "message": "Stopping workflow...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Stop the workflow
        workflow_manager.stop_inspection_workflow()
        
        # Notify clients that we've stopped
        await websocket_manager.broadcast({
            "type": "workflow_status",
            "status": "stopped",
            "message": "Workflow stopped successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("Workflow stopped successfully")
        
        return {
            "status": "success",
            "message": "Workflow stopped successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error stopping workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Notify clients about the error
        await websocket_manager.broadcast({
            "type": "workflow_status",
            "status": "error",
            "message": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@api_router.post("/realtime/workflow/export")
async def export_session_report():
    """Export session report"""
    try:
        if not workflow_manager:
            error_msg = "Workflow manager not initialized"
            logger.error(error_msg)
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "message": error_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Note: The export_session_report method needs a filename parameter
        # For now, we'll use a default filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"session_report_{timestamp}.json"
        report_path = workflow_manager.export_session_report(filename)
        
        if report_path:
            return {
                "success": True, 
                "report_path": report_path,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to export session report"
            )
    except Exception as e:
        logger.error(f"Error exporting session report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@api_router.websocket("/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            try:
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
                elif data.get('type') == 'get_state':
                    if workflow_manager:
                        state = workflow_manager.get_current_state()
                        await websocket.send_json({
                            'type': 'state_update',
                            'data': state
                        })
                    else:
                        await websocket.send_json({
                            'type': 'error',
                            'data': {'message': 'Workflow manager not initialized'}
                        })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    'type': 'error',
                    'data': {'message': str(e)}
                })
    
    finally:
        websocket_manager.disconnect(websocket)

# NOTE: api_router is included at the bottom of the file, after all routes are defined

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    if client is not None:
        client.close()


### BEGIN: ADDITIONAL HELPERS & UNIQUE ENDPOINTS
import sys
import io

# Increase CSV field size limit to handle large JSON data
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

# Ensure CSV file exists with headers
if not CSV_LOG_PATH.exists():
    with open(CSV_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id","timestamp","mode","filename","is_defective","confidence_score","defect_count","defects","annotated_relpath"])

def _ensure_csv_headers():
    """Ensure CSV file exists with proper headers."""
    try:
        if not CSV_LOG_PATH.exists() or os.path.getsize(CSV_LOG_PATH) == 0:
            CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CSV_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "timestamp", "mode", "filename", 
                    "is_defective", "confidence_score", 
                    "defect_count", "defects", "annotated_relpath"
                ])
        return True
    except Exception as e:
        logger.error(f"Error ensuring CSV headers: {e}")
        return False

def _append_csv(row: Dict[str, Any]) -> bool:
    """Append a new inspection record to the CSV file."""
    try:
        if not _ensure_csv_headers():
            return False
            
        with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            defect_count = 0
            if "defects" in row:
                if isinstance(row["defects"], list):
                    defect_count = len(row["defects"])
                elif isinstance(row["defects"], str):
                    try:
                        defects_data = json.loads(row["defects"])
                        if isinstance(defects_data, list):
                            defect_count = len(defects_data)
                    except:
                        pass
            
            writer.writerow([
                str(row.get("id", "")),
                str(row.get("timestamp", "")),
                str(row.get("mode", "manual")),
                str(row.get("filename", "")),
                str(row.get("is_defective", "false")).lower(),
                str(float(row.get("confidence_score", 0.0))),
                str(int(row.get("defect_count", 0))),
                str(defect_count),
                str(row.get("annotated_relpath", ""))
            ])
        return True
    except Exception as e:
        logger.error(f"Error appending to CSV: {e}")
        return False

# --- Unique endpoints (no duplicates on api_router) ---

@api_router.post("/pcb/test-data")
async def create_test_data():
    """Create some test inspection data for development"""
    try:
        test_records = [
            {
                "id": "test-1",
                "timestamp": "2025-08-22T20:00:00.000000",
                "mode": "api",
                "filename": "test_pcb_1.jpg",
                "is_defective": "true",
                "confidence_score": "0.85",
                "defect_count": "3",
                "defects": "3",
                "annotated_relpath": "/api/pcb/defective/defective_pcb_20250822_183022.jpg"
            },
            {
                "id": "test-2", 
                "timestamp": "2025-08-22T19:30:00.000000",
                "mode": "api",
                "filename": "test_pcb_2.jpg",
                "is_defective": "false",
                "confidence_score": "0.92",
                "defect_count": "0",
                "defects": "0",
                "annotated_relpath": "/api/pcb/defective/defective_pcb_20250822_181751.jpg"
            }
        ]
        for record in test_records:
            _append_csv(record)
        return {"message": "Test data created successfully", "count": len(test_records)}
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/pcb/test")
async def test_endpoint():
    """Simple test endpoint to verify backend is working"""
    return {"message": "Backend is working", "timestamp": datetime.utcnow().isoformat()}

@api_router.get("/pcb/inspections.csv")
async def api_download_inspections_csv():
    """Send the CSV file for direct download."""
    try:
        if not _ensure_csv_headers():
            raise HTTPException(status_code=500, detail="Failed to initialize CSV file")
            
        if not CSV_LOG_PATH.exists() or os.path.getsize(CSV_LOG_PATH) == 0:
            headers = ["id", "timestamp", "mode", "filename", 
                     "is_defective", "confidence_score", 
                     "defect_count", "defects", "annotated_relpath"]
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(headers)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=inspections.csv"}
            )
        
        valid_rows = []
        with open(CSV_LOG_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if not row.get('id') or not row.get('filename'):
                    continue
                if 'defects' in row and isinstance(row['defects'], str):
                    try:
                        defects_data = json.loads(row['defects'])
                        row['defects'] = str(len(defects_data)) if isinstance(defects_data, list) else "0"
                    except:
                        row['defects'] = "0"
                valid_rows.append(row)
        
        output = io.StringIO()
        default_fieldnames = ["id", "timestamp", "mode", "filename", 
                            "is_defective", "confidence_score", 
                            "defect_count", "defects", "annotated_relpath"]
        writer = csv.DictWriter(output, fieldnames=fieldnames if fieldnames else default_fieldnames)
        writer.writeheader()
        for row in valid_rows:
            valid_row = {field: row.get(field, "") for field in (fieldnames if fieldnames else default_fieldnames)}
            writer.writerow(valid_row)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=inspections.csv",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )
    except Exception as e:
        logger.error(f"Error in api_download_inspections_csv: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

@api_router.get("/realtime/camera/preview")
async def api_realtime_preview():
    """Return raw camera preview with timestamp"""
    try:
        if not camera_manager:
            return generate_placeholder_image("Camera manager not initialized")
        if not camera_manager.is_camera_connected():
            return generate_placeholder_image("Camera not connected")
        frame = camera_manager.capture_single_image()
        if frame is None:
            return generate_placeholder_image("Failed to capture frame")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return create_image_response(frame)
    except Exception as e:
        logger.error(f"Error in preview: {e}")
        return generate_placeholder_image("Error: " + str(e))

# --- Helper functions for image responses ---

def generate_placeholder_image(message):
    """Generate a placeholder image with error message"""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, message, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    _, buf = cv2.imencode(".jpg", img)
    return StreamingResponse(
        BytesIO(buf.tobytes()),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

def create_image_response(image):
    """Create a streaming response from an image"""
    _, buffer = cv2.imencode('.jpg', image, [
        cv2.IMWRITE_JPEG_QUALITY, 80,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1
    ])
    return StreamingResponse(
        BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

### END: ADDITIONAL HELPERS & UNIQUE ENDPOINTS

# Include the router in the main app (MUST be after all @api_router routes are defined)
app.include_router(api_router)

@app.get("/health")
async def health():
    return {"ok": True}
