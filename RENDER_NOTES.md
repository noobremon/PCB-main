# 📝 Render Deployment Notes

## Important Considerations for Render Deployment

---

## 🎯 Key Differences from Local Development

### 1. **Filesystem is Ephemeral**
- Files written at runtime are lost on redeploy
- Don't store uploaded PCB images on local disk
- **Solution:** Use cloud storage (AWS S3, Cloudinary, etc.)

### 2. **Cold Starts (Free Tier)**
- Services sleep after 15 minutes of inactivity
- First request takes 30-60 seconds to wake up
- **Solution:** Upgrade to Starter ($7/mo) or use a ping service

### 3. **Build Time Limits**
- Free tier: 15 minutes build timeout
- Heavy ML libraries (OpenCV, scikit-learn) take time
- **Solution:** Optimize dependencies or use pre-built wheels

### 4. **Memory Constraints**
- Free tier: 512 MB RAM
- Starter: 512 MB RAM
- Standard: 2 GB RAM
- **Consider:** ML model can be memory-intensive

---

## 🔧 Recommended Optimizations

### Backend (`server.py`)

#### 1. Disable Camera Features for Cloud
Camera hardware won't work in cloud environment:

```python
# Around line 70-80, add environment check
import os
ENABLE_CAMERA = os.environ.get('ENABLE_CAMERA', 'false').lower() == 'true'

# Then conditionally initialize camera_manager
if ENABLE_CAMERA:
    camera_manager = initialize_camera()
else:
    camera_manager = None
    logger.info("Camera features disabled for cloud deployment")
```

#### 2. Use Cloud Storage for Images

Replace local file storage with cloud storage:

```python
# Install: boto3 for AWS S3
import boto3

S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
s3_client = boto3.client('s3')

def save_defective_image(image, filename):
    # Upload to S3 instead of local disk
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    
    s3_client.upload_fileobj(
        buffer,
        S3_BUCKET,
        f'defective/{filename}'
    )
    return f'https://{S3_BUCKET}.s3.amazonaws.com/defective/{filename}'
```

#### 3. Lazy Load ML Model

Don't build model on startup:

```python
# Build model on first request instead of startup
pcb_inspector = None

async def get_inspector():
    global pcb_inspector
    if pcb_inspector is None:
        pcb_inspector = IndustrialPCBInspector(...)
        await ensure_reference_model()
    return pcb_inspector
```

---

## 🗄️ Database Considerations

### MongoDB Atlas Setup

**Connection String Format:**
```
mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

**Important Settings:**
1. **Network Access:** Add `0.0.0.0/0` to IP whitelist
2. **Database User:** Create with read/write permissions
3. **Connection Options:** Include `retryWrites=true`

**Environment Variable:**
```bash
MONGO_URL=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
```

---

## 📦 Dependency Management

### For Faster Builds

Create a slimmed-down `requirements-render.txt`:

```txt
# Core FastAPI
fastapi==0.110.1
uvicorn[standard]==0.30.1
python-dotenv==1.0.1
python-multipart==0.0.9

# Database
motor==3.4.0
pymongo==4.7.2

# Data processing
numpy==1.26.0
pandas==2.2.0

# Image processing (lighter alternatives)
opencv-python-headless==4.9.0.80  # Headless version (smaller)
pillow>=10.0.0
scikit-image==0.22.0
scikit-learn==1.4.2
scipy==1.12.0

# Auth & Security
pyjwt>=2.10.1
cryptography>=42.0.8

# Other
requests>=2.31.0
pydantic==2.7.1
```

Update `render.yaml` to use this:
```yaml
buildCommand: "pip install -r backend/requirements-render.txt"
```

---

## 🚀 Performance Optimizations

### 1. Enable Response Caching

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())

@api_router.get("/api/statistics")
@cache(expire=60)  # Cache for 60 seconds
async def get_statistics():
    # ... your code
```

### 2. Use Async Database Queries

Already implemented with Motor, but ensure all DB calls use `await`:

```python
# Good (async)
results = await db.inspections.find().to_list(100)

# Bad (sync - blocks event loop)
results = list(db.inspections.find())
```

### 3. Limit Inspection History

Don't load entire history:

```python
@api_router.get("/api/history")
async def get_history(limit: int = 50, skip: int = 0):
    results = await db.inspections.find() \
        .sort("timestamp", -1) \
        .skip(skip) \
        .limit(limit) \
        .to_list(limit)
    return results
```

---

## 🌐 CORS Configuration

### Production CORS Settings

Update `server.py` CORS middleware:

```python
# Get frontend URL from environment
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "https://*.onrender.com",  # Allow all Render subdomains
        "http://localhost:3000",  # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Add to Render environment variables:
```
FRONTEND_URL=https://your-frontend.onrender.com
```

---

## 📊 Monitoring & Logging

### 1. Structured Logging for Render

Render captures stdout/stderr:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        return json.dumps(log_obj)

# Setup logger
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

### 2. Health Check Endpoint

Already exists but enhance it:

```python
@app.get("/health")
async def health():
    # Check database connection
    try:
        await db.command('ping')
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "version": "1.0.0"
    }
```

---

## 🔐 Security Enhancements

### 1. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@api_router.post("/api/inspect")
@limiter.limit("10/minute")  # 10 requests per minute
async def inspect_pcb(request: Request, file: UploadFile):
    # ... your code
```

### 2. File Upload Validation

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

async def validate_upload(file: UploadFile):
    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    await file.seek(0)  # Reset file pointer
    return True
```

---

## 💾 Data Persistence Strategy

### Option 1: AWS S3

```python
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

async def save_to_s3(image_data, filename):
    try:
        s3.put_object(
            Bucket=os.environ['S3_BUCKET'],
            Key=f'defective/{filename}',
            Body=image_data,
            ContentType='image/jpeg'
        )
        return f"https://{os.environ['S3_BUCKET']}.s3.amazonaws.com/defective/{filename}"
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        return None
```

### Option 2: Cloudinary

```python
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.environ['CLOUDINARY_CLOUD_NAME'],
    api_key=os.environ['CLOUDINARY_API_KEY'],
    api_secret=os.environ['CLOUDINARY_API_SECRET']
)

async def save_to_cloudinary(image_data, filename):
    result = cloudinary.uploader.upload(
        image_data,
        folder="pcb-defective",
        public_id=filename
    )
    return result['secure_url']
```

---

## 🔄 CI/CD Best Practices

### Auto-Deploy on Git Push

Render automatically deploys on push to main branch.

**To disable:**
1. Go to service settings
2. Uncheck "Auto-Deploy"

**Branch-based deployment:**
```yaml
# render.yaml
services:
  - type: web
    name: pcb-backend-staging
    branch: develop  # Deploy develop branch
    
  - type: web
    name: pcb-backend-production
    branch: main  # Deploy main branch
```

---

## 🐛 Common Issues and Solutions

### Issue: Build Fails with "Out of Memory"

**Solution:**
1. Use opencv-python-headless instead of opencv-python
2. Upgrade to Standard plan (2GB RAM)
3. Remove unused dependencies

### Issue: Cold Start Too Slow

**Solutions:**
1. Upgrade to paid plan ($7/mo)
2. Use external ping service (cron-job.org, UptimeRobot)
3. Reduce startup tasks (lazy load models)

### Issue: WebSocket Connection Fails

**Solution:** Render supports WebSockets, but check:
```python
# Ensure WebSocket endpoint is correct
ws = new WebSocket('wss://your-backend.onrender.com/ws/inspection')
# Use wss:// not ws:// for HTTPS
```

### Issue: Static Files Not Found

**Solution:** Mount static files correctly:
```python
app.mount("/static", StaticFiles(directory="static"), name="static")
```

### Issue: MongoDB Connection Timeout

**Solutions:**
1. Check IP whitelist includes `0.0.0.0/0`
2. Verify connection string has `?retryWrites=true`
3. Increase timeout in connection string:
   ```
   mongodb+srv://.../?retryWrites=true&w=majority&serverSelectionTimeoutMS=5000
   ```

---

## 📈 Scaling Considerations

### Horizontal Scaling

Render supports multiple instances:
```yaml
services:
  - type: web
    name: pcb-backend
    numInstances: 2  # Run 2 instances
```

**Note:** With multiple instances, use:
- External session storage (Redis)
- MongoDB for shared state
- S3/Cloudinary for file storage

### Vertical Scaling

Upgrade instance types:
- **Free:** 512 MB RAM, shared CPU
- **Starter:** 512 MB RAM, shared CPU ($7/mo)
- **Standard:** 2 GB RAM, shared CPU ($25/mo)
- **Pro:** 4 GB RAM, dedicated CPU ($85/mo)

---

## 🎯 Production Checklist

- [ ] MongoDB Atlas cluster created and accessible
- [ ] Environment variables configured (no hardcoded secrets)
- [ ] CORS configured with frontend URL
- [ ] Health check endpoint working
- [ ] Camera features disabled for cloud
- [ ] File uploads go to cloud storage (not local disk)
- [ ] Error logging properly configured
- [ ] Rate limiting enabled
- [ ] File upload validation in place
- [ ] Database indexes created for performance
- [ ] SSL/HTTPS enabled (automatic on Render)
- [ ] Monitoring/alerting setup
- [ ] Backup strategy for MongoDB
- [ ] Documentation updated with deployment URLs

---

## 📞 Support & Resources

- **Render Docs:** https://render.com/docs
- **Render Community:** https://community.render.com
- **FastAPI Deployment:** https://fastapi.tiangolo.com/deployment/
- **MongoDB Atlas:** https://docs.atlas.mongodb.com

---

**Last Updated:** February 2026

**Questions?** Check [DEPLOYMENT.md](DEPLOYMENT.md) or open an issue.
