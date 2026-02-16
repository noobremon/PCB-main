# 🔬 Industrial PCB AOI System

**Automated Optical Inspection System for Printed Circuit Boards**

A production-grade, real-time PCB inspection system using computer vision and machine learning to detect manufacturing defects in printed circuit boards. This system supports both manual inspection via image upload and automated real-time inspection with industrial camera integration.

---

## ✨ Key Features

- 🎯 **Real-Time Inspection** - Automated workflow with industrial camera support (Baumer SDK, OpenCV)
- 🖼️ **Manual Inspection** - Upload and analyze PCB images on demand
- 🤖 **Machine Learning Detection** - Advanced defect detection using scikit-learn, OpenCV, and scikit-image
- 📊 **Quality Metrics** - Comprehensive statistics, quality scores, and confidence levels
- 🔄 **WebSocket Integration** - Real-time updates and live camera preview
- 📈 **Historical Tracking** - Complete inspection history with MongoDB storage
- 🎨 **Modern UI** - Responsive React dashboard with Tailwind CSS
- 🏭 **Industrial Grade** - Production-ready with error handling, logging, and fallback mechanisms
- 📸 **Multiple Camera Support** - Compatible with industrial cameras and standard webcams
- 🔍 **Advanced CV Algorithms** - Multi-stage defect detection with morphological operations

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+** - Core language
- **FastAPI** - High-performance async web framework
- **OpenCV** - Computer vision and image processing
- **scikit-learn** - Machine learning for anomaly detection
- **scikit-image** - Advanced image analysis algorithms
- **MongoDB + Motor** - Async database with MongoDB driver
- **NumPy & SciPy** - Scientific computing
- **Uvicorn** - ASGI server

### Frontend
- **React 19** - Modern UI library
- **Axios** - HTTP client for API communication
- **Tailwind CSS** - Utility-first styling
- **React Router** - Client-side routing
- **WebSocket** - Real-time communication

### Optional Industrial Features
- **Baumer GenICam SDK** - Support for industrial cameras
- **pypylon** - Basler camera support
- **pyueye** - IDS camera support

---

## 📁 Project Structure

```
PCB-main/
│
├── backend/                          # Backend API server
│   ├── server.py                     # FastAPI application & API endpoints
│   ├── requirements.txt              # Python dependencies
│   ├── camera_config.json            # Camera settings
│   ├── workflow_config.json          # Inspection workflow config
│   ├── inspections.csv               # Inspection records
│   ├── inspection_logs.csv           # Detailed logs
│   ├── defective_storage/            # Stored defective PCB images
│   └── inspection_results/           # Inspection result files
│
├── frontend/                         # React frontend application
│   ├── src/
│   │   ├── App.js                    # Main React component
│   │   ├── App.css                   # Component styles
│   │   └── index.js                  # React entry point
│   ├── public/
│   │   └── index.html                # HTML template
│   ├── package.json                  # Node dependencies
│   ├── craco.config.js               # Create React App override
│   └── tailwind.config.js            # Tailwind CSS configuration
│
├── dataset/                          # Training and test datasets
│   ├── good/                         # Good PCB samples
│   ├── defective/                    # Defective PCB samples
│   ├── raw/                          # Raw dataset
│   │   ├── good/
│   │   └── defective/
│   └── marked/                       # Annotated dataset
│       ├── images/
│       └── annotations/
│
├── test_images/                      # Test images for validation
│   └── check/
│
├── defective_storage/                # Root-level defective storage
│   └── realtime/                     # Real-time inspection results
│
├── test_results/                     # Test run results
│
├── enhanced_pcb_inspection.py        # Core inspection engine
├── camera_integration.py             # Camera abstraction layer
├── realtime_workflow.py              # Workflow state machine
├── consistency_verifier.py           # Data validation utilities
│
├── camera_config.json                # Root camera configuration
├── workflow_config.json              # Root workflow configuration
├── requirements-industrial.txt       # Optional industrial dependencies
│
└── README.md                         # This file
```

### 🧩 Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                      │
│  - Dashboard UI                                         │
│  - Manual Inspection Upload                             │
│  - Real-time Camera Preview                             │
│  - Statistics & History Display                         │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/WebSocket
┌──────────────────▼──────────────────────────────────────┐
│                   FastAPI Backend                       │
│  - REST API Endpoints                                   │
│  - WebSocket Handler                                    │
│  - File Upload Management                               │
└──────────────┬────────────────────┬─────────────────────┘
               │                    │
      ┌────────▼────────┐  ┌────────▼──────────┐
      │ MongoDB Storage │  │ Workflow Manager  │
      │  - Inspections  │  │  - State Machine  │
      │  - Statistics   │  │  - Auto Mode      │
      └─────────────────┘  └────────┬──────────┘
                                    │
                           ┌────────▼──────────┐
                           │ Camera Manager    │
                           │  - Baumer SDK     │
                           │  - OpenCV Fallback│
                           └────────┬──────────┘
                                    │
                           ┌────────▼──────────┐
                           │ PCB Inspector     │
                           │  - CV Algorithms  │
                           │  - ML Detection   │
                           │  - Quality Score  │
                           └───────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- **Python** 3.8 or higher
- **Node.js** 16+ and npm/yarn
- **MongoDB** (local or cloud instance)
- **Camera** (optional, for real-time mode)

### 1. Clone Repository

```bash
git clone https://github.com/noobremon/PCB-main.git
cd PCB-main
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Optional: Install industrial camera support
pip install -r requirements-industrial.txt
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
# or
yarn install

cd ..
```

### 4. Dataset Preparation

Place your PCB training images in the appropriate directories:

```bash
dataset/
├── good/           # Add good PCB images here
└── defective/      # Add defective PCB images here
```

---

## ⚙️ Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# MongoDB Configuration
MONGO_URL=mongodb://localhost:27017
DB_NAME=pcb_inspection

# Optional: Advanced Settings
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=10485760
```

### Frontend Environment

Create a `.env` file in the `frontend/` directory:

```env
# Backend API URL
REACT_APP_BACKEND_URL=http://localhost:8000
```

---

## 🏃 Running Locally

### Start Backend Server

```bash
# From project root, with venv activated
cd backend
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at **http://localhost:8000**

### Start Frontend Development Server

```bash
# In a new terminal
cd frontend
npm start
# or
yarn start
```

The frontend will be available at **http://localhost:3000**

### Access the Application

Open your browser and navigate to:
- **Frontend UI**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc

---

## 📦 Build & Deployment

### Backend Production Build

```bash
# Install production dependencies
pip install -r backend/requirements.txt

# Run with production settings
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

**Production Considerations:**
- Use a process manager like **Gunicorn** or **Supervisor**
- Set up reverse proxy with **Nginx**
- Enable HTTPS with SSL certificates
- Configure CORS for your production domain in `server.py`

### Frontend Production Build

```bash
cd frontend

# Build optimized production bundle
npm run build
# or
yarn build
```

The production-ready static files will be in `frontend/build/`

**Deployment Options:**
- **Static Hosting**: Deploy `build/` to Netlify, Vercel, or AWS S3
- **Docker**: Create Docker containers for both frontend and backend
- **Traditional Hosting**: Serve `build/` with Nginx or Apache

### Docker Deployment (Recommended)

Create `Dockerfile` for backend:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `Dockerfile` for frontend:

```dockerfile
FROM node:16-alpine as build

WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 🔌 API Endpoints

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/` | Health check |
| `GET` | `/api/status` | Get system status |
| `GET` | `/api/pcb/stats` | Get inspection statistics |

### Manual Inspection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/pcb/train` | Train inspection model |
| `POST` | `/api/pcb/inspect` | Inspect uploaded PCB image |
| `GET` | `/api/pcb/inspections` | Get inspection history |
| `GET` | `/api/pcb/result/{filename}` | Get result image |
| `GET` | `/api/pcb/defective/{filename}` | Get defective PCB image |

### Real-Time Inspection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/pcb/realtime/available-cameras` | List available cameras |
| `POST` | `/api/realtime/camera/connect` | Connect to camera |
| `POST` | `/api/realtime/camera/disconnect` | Disconnect camera |
| `POST` | `/api/realtime/workflow/start` | Start inspection workflow |
| `POST` | `/api/realtime/workflow/trigger` | Trigger single inspection |
| `POST` | `/api/realtime/workflow/stop` | Stop workflow |
| `GET` | `/api/realtime/workflow/state` | Get current workflow state |
| `GET` | `/api/realtime/workflow/stats` | Get workflow statistics |

### WebSocket

| Type | Endpoint | Description |
|------|----------|-------------|
| `WebSocket` | `/api/realtime/ws` | Real-time updates stream |

---

## 💡 Usage Examples

### Manual Inspection via API

```bash
# Train the model
curl -X POST http://localhost:8000/api/pcb/train

# Inspect an image
curl -X POST http://localhost:8000/api/pcb/inspect \
  -F "file=@pcb_sample.jpg"
```

### Using Python Client

```python
import requests

# Train model
response = requests.post('http://localhost:8000/api/pcb/train')
print(response.json())

# Inspect image
with open('pcb_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/pcb/inspect',
        files=files
    )
    result = response.json()
    print(f"Defective: {result['is_defective']}")
    print(f"Quality Score: {result['quality_score']}")
    print(f"Defects Found: {len(result['defects'])}")
```

### Real-Time Workflow

```python
import requests

# Connect to camera
response = requests.post(
    'http://localhost:8000/api/realtime/camera/connect',
    json={'camera_id': 0, 'camera_type': 'opencv'}
)

# Start automatic inspection workflow
response = requests.post(
    'http://localhost:8000/api/realtime/workflow/start',
    json={'auto_mode': True}
)

# Check workflow state
response = requests.get('http://localhost:8000/api/realtime/workflow/state')
print(response.json())
```

---

## 📸 Screenshots

> **Note**: Add screenshots of your application here

### Dashboard View
![Dashboard](docs/screenshots/dashboard.png)

### Manual Inspection
![Manual Inspection](docs/screenshots/manual-inspection.png)

### Real-Time Inspection
![Real-Time Mode](docs/screenshots/realtime-mode.png)

### Defect Detection Results
![Results](docs/screenshots/results.png)

---

## 🔧 Configuration

### Camera Configuration (`camera_config.json`)

```json
{
  "preferred_camera": "opencv",
  "opencv_camera_index": 0,
  "resolution": [1280, 720],
  "fps": 30,
  "quality": 80
}
```

### Workflow Configuration (`workflow_config.json`)

```json
{
  "capture_delay": 1.0,
  "inspection_timeout": 30.0,
  "result_display_time": 10.0,
  "min_quality_score": 50.0,
  "auto_save_defective": true
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** and commit
   ```bash
   git commit -m "Add: amazing new feature"
   ```
4. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Coding Standards

- Follow **PEP 8** for Python code
- Use **ESLint** configuration for JavaScript
- Write meaningful commit messages
- Add comments for complex logic
- Include docstrings for functions and classes
- Write unit tests for new features

### Reporting Issues

- Use GitHub Issues to report bugs
- Include system information (OS, Python version, etc.)
- Provide steps to reproduce the issue
- Attach relevant logs or screenshots

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 PCB Inspection Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🚀 Future Improvements

### Planned Features

- [ ] **Deep Learning Integration** - Add CNN-based defect classification models (TensorFlow/PyTorch)
- [ ] **Multi-Camera Support** - Simultaneous inspection with multiple cameras
- [ ] **3D Inspection** - Support for 3D PCB inspection with depth cameras
- [ ] **Cloud Integration** - AWS S3/Azure Blob storage for images
- [ ] **Advanced Analytics** - Defect trend analysis and predictive maintenance
- [ ] **Mobile App** - React Native app for remote monitoring
- [ ] **User Management** - Role-based access control and authentication
- [ ] **Export Reports** - Generate PDF/Excel inspection reports
- [ ] **Email Notifications** - Alert system for critical defects
- [ ] **REST API Versioning** - Support multiple API versions
- [ ] **GraphQL API** - Alternative to REST for flexible queries
- [ ] **Docker Compose** - One-command deployment setup
- [ ] **Kubernetes Support** - Production-grade orchestration configs
- [ ] **Performance Monitoring** - Integration with Prometheus/Grafana
- [ ] **Automated Testing** - Comprehensive unit and integration tests
- [ ] **CI/CD Pipeline** - GitHub Actions for automated testing and deployment

### Enhancement Ideas

- **AI-Powered Suggestions** - Recommend fixes for detected defects
- **Barcode/QR Scanning** - Automated PCB identification
- **Real-Time Alerts** - Push notifications via Firebase/WebPush
- **Historical Comparison** - Compare current PCB with previous versions
- **Batch Processing** - Inspect multiple images in parallel
- **Custom Defect Types** - User-defined defect categories and training
- **Calibration Tools** - Camera calibration utilities
- **Performance Optimization** - GPU acceleration for CV algorithms
- **Localization** - Multi-language support

---

## 📞 Support & Contact

For questions, issues, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/noobremon/PCB-main/issues)
- **Email**: support@pcb-inspection.com
- **Documentation**: [Wiki](https://github.com/noobremon/PCB-main/wiki)

---

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **FastAPI** - Modern web framework
- **React** - UI library
- **Tailwind CSS** - Styling framework
- **scikit-learn** - Machine learning toolkit
- **scikit-image** - Image processing library

---

## 📊 Project Status

**Status**: ✅ Active Development

**Version**: 1.0.0

**Last Updated**: February 2026

---

<div align="center">

**Built with ❤️ by the PCB Inspection Team**

[⬆ Back to Top](#-industrial-pcb-aoi-system)

</div>
