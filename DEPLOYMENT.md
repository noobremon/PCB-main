# 🚀 Deployment Guide for Render

## Project: PCB Automated Optical Inspection System

This guide will help you successfully deploy your PCB inspection application to Render.

---

## 📋 Prerequisites

Before deploying, ensure you have:

1. ✅ A Render account (sign up at https://render.com)
2. ✅ A MongoDB Atlas account (or other cloud MongoDB service) - https://www.mongodb.com/cloud/atlas
3. ✅ Git repository pushed to GitHub/GitLab/Bitbucket
4. ✅ All code committed and pushed

**Note:** Render uses **native buildpacks** (not Docker), so no Dockerfile needed! Render automatically detects your Python and Node.js apps and builds them.

---

## 🗄️ Step 1: Setup MongoDB Database

### Option A: MongoDB Atlas (Recommended)

1. Go to https://www.mongodb.com/cloud/atlas
2. Create a free cluster (M0 Sandbox)
3. Create a database user with username and password
4. Whitelist IP address: `0.0.0.0/0` (to allow all connections)
5. Get your connection string - it will look like:
   ```
   mongodb+srv://username:password@cluster.mongodb.net/
   ```

### Option B: Use Render's MongoDB Add-on

Render doesn't offer MongoDB directly, so Atlas is recommended.

---

## 🔧 Step 2: Deploy Backend as Web Service

1. **Login to Render Dashboard**
   - Go to https://dashboard.render.com
   - Sign in with your GitHub/GitLab account

2. **Create New Web Service**
   - Click **"New +"** button (top right)
   - Select **"Web Service"**
   - Connect your Git repository if not already connected
   - Select your repository: **PCB-main**

3. **Configure Backend Service**
   
   Fill in these settings:
   
   | Setting | Value |
   |---------|-------|
   | **Name** | `pcb-inspection-backend` (or your choice) |
   | **Region** | Oregon (US West) or closest to you |
   | **Branch** | `main` |
   | **Root Directory** | `backend` ⚠️ **Important!** |
   | **Runtime** | Python 3 (auto-detected) |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn server:app --host 0.0.0.0 --port $PORT` |
   | **Instance Type** | Free or Starter ($7/month for no cold starts) |

4. **Add Environment Variables**
   
   Scroll down and click **"Advanced"** → **"Add Environment Variable"**
   
   Add these variables one by one:
   
   | Key | Value |
   |-----|-------|
   | `MONGO_URL` | Your MongoDB connection string from Step 1 |
   | `DB_NAME` | `pcb_inspection` |
   | `LOG_LEVEL` | `INFO` |
   | `MAX_UPLOAD_SIZE` | `10485760` |
   | `PYTHON_VERSION` | `3.11.0` |

5. **Create Web Service**
   - Click **"Create Web Service"** button at the bottom
   - Render will start building and deploying (takes 5-10 minutes)
   - Watch the logs to monitor progress

6. **Save Your Backend URL**
   - Once deployed, you'll see your backend URL at the top
   - It will look like: `https://pcb-inspection-backend.onrender.com`
   - **Copy this URL** - you'll need it for the frontend!

---

## 🎨 Step 3: Deploy Frontend as Static Site

1. **Create New Static Site**
   - In Render Dashboard, click **"New +"** button (top right)
   - Select **"Static Site"**
   - Select the same repository: **PCB-main**

2. **Configure Static Site**
   
   Fill in these settings:
   
   | Setting | Value |
   |---------|-------|
   | **Name** | `pcb-inspection-frontend` (or your choice) |
   | **Branch** | `main` |
   | **Root Directory** | `frontend` ⚠️ **Important!** |
   | **Build Command** | `npm install && npm run build` |
   | **Publish Directory** | `build` |

3. **Add Environment Variable**
   
   Click **"Advanced"** → **"Add Environment Variable"**
   
   Add this variable:
   
   | Key | Value |
   |-----|-------|
   | `REACT_APP_BACKEND_URL` | Your backend URL from Step 2<br>(e.g., `https://pcb-inspection-backend.onrender.com`) |
   
   ⚠️ **Important:** Make sure to use the exact URL (with `https://`) and **no trailing slash**

4. **Create Static Site**
   - Click **"Create Static Site"** button at the bottom
   - Render will build and deploy (takes 3-5 minutes)
   - Watch the build logs to monitor progress

5. **Save Your Frontend URL**
   - Once deployed, you'll see your frontend URL
   - It will look like: `https://pcb-inspection-frontend.onrender.com`
   - **This is your live app URL!** 🎉

---

## 🔄 Step 4: Update CORS Settings

After deployment, you may need to update CORS settings in your backend:

1. Open [backend/server.py](backend/server.py)
2. Update the CORS middleware (around line 80):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pcb-inspection-frontend.onrender.com",  # Your frontend URL
        "http://localhost:3000"  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

3. Commit and push changes
4. Render will auto-redeploy

---

## ✅ Step 5: Verify Deployment

### Test Backend
1. Visit: `https://your-backend-url.onrender.com/health`
2. Should return: `{"ok": true}`

3. Visit: `https://your-backend-url.onrender.com/api/`
4. Should return: `{"message": "PCB Automated Optical Inspection System API"}`

### Test Frontend
1. Visit: `https://your-frontend-url.onrender.com`
2. You should see the PCB inspection dashboard
3. Try uploading an image for inspection

---

## 📁 Step 6: Initialize Dataset (Optional)

For ML model to work, you need sample PCB images:

### Option 1: Upload via API
- Use the upload endpoints to add good and defective PCB samples

### Option 2: Pre-populate Dataset
1. Add sample images to GitHub:
   - `dataset/good/` - Good PCB images
   - `dataset/defective/` - Defective PCB images
2. Push to repository
3. Render will include them in deployment

---

## 🐛 Troubleshooting

### Issue: Backend fails to start

**Solution:** Check build logs in Render dashboard
- Ensure all dependencies in `requirements.txt` are compatible
- Check MongoDB connection string is correct
- Verify environment variables are set

### Issue: Frontend can't connect to backend

**Solution:**
1. Check `REACT_APP_BACKEND_URL` is correct
2. Verify CORS settings in backend
3. Ensure backend is running (check health endpoint)

### Issue: MongoDB connection timeout

**Solution:**
1. Verify MongoDB Atlas IP whitelist includes `0.0.0.0/0`
2. Check username/password in connection string
3. Ensure database user has correct permissions

### Issue: Build fails due to memory

**Solution:**
- Upgrade to a paid Render plan (more memory)
- Or optimize dependencies in `requirements.txt`
- Remove unused packages

### Issue: Cold starts (free tier)

**Note:** Free tier services sleep after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- Upgrade to paid plan to keep service active 24/7

---

## 💰 Cost Estimate

### Free Tier Option:
- Backend: Free tier (with cold starts)
- Frontend: Free tier
- MongoDB Atlas: Free tier (M0)
- **Total: $0/month**

### Paid Option:
- Backend: Starter ($7/month)
- Frontend: Free tier
- MongoDB Atlas: Free tier (M0)
- **Total: $7/month**

### Production Option:
- Backend: Standard ($25/month)
- Frontend: Static site with CDN ($1/month)
- MongoDB Atlas: M10 ($9/month)
- **Total: $35/month**

---

## 🔐 Security Best Practices

1. **Environment Variables**
   - Never commit `.env` files to Git
   - Use Render's environment variable management

2. **MongoDB Security**
   - Use strong passwords
   - Enable IP whitelisting when possible
   - Regularly rotate credentials

3. **API Security**
   - Update CORS to allow only your frontend domain
   - Implement rate limiting for production
   - Consider adding authentication/authorization

4. **HTTPS**
   - Render provides free SSL certificates
   - All traffic is encrypted by default

---

## 🚀 Continuous Deployment

Render automatically deploys on Git push:

1. Make changes to your code
2. Commit and push to your repository
3. Render detects changes and auto-deploys
4. Monitor deployment in Render dashboard

To disable auto-deploy:
- Go to service settings → Auto-Deploy → Off

---

## 📊 Monitoring

### Render Dashboard
- View logs in real-time
- Monitor resource usage (CPU, memory)
- Check deployment history
- Set up alerts for service failures

### Application Logs
- Backend logs available in Render dashboard
- Filter by severity level
- Download logs for analysis

---

## 🔄 Rollback

If deployment fails:

1. Go to Render Dashboard
2. Click on your service
3. Navigate to "Events" tab
4. Find previous successful deployment
5. Click "Rollback to this version"

---

## 📞 Support Resources

- **Render Documentation:** https://render.com/docs
- **Render Community:** https://community.render.com
- **MongoDB Atlas Docs:** https://docs.atlas.mongodb.com
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **React Docs:** https://react.dev

---

## 🎯 Next Steps After Deployment

1. **Test all features:**
   - Manual PCB inspection
   - Real-time camera inspection (if camera available)
   - View inspection history
   - Check statistics dashboard

2. **Optimize Performance:**
   - Monitor response times
   - Optimize image processing
   - Consider caching strategies

3. **Add Monitoring:**
   - Setup uptime monitoring (UptimeRobot, Pingdom)
   - Configure error tracking (Sentry)
   - Setup log aggregation

4. **Scale as needed:**
   - Monitor resource usage
   - Upgrade instance types if needed
   - Consider Redis for caching

---

## ✨ Deployment Checklist

- [ ] MongoDB Atlas cluster created
- [ ] Database user created with password
- [ ] IP whitelist configured (0.0.0.0/0)
- [ ] Backend deployed to Render
- [ ] Backend environment variables configured
- [ ] Backend health check passes
- [ ] Frontend deployed to Render
- [ ] Frontend environment variable set (REACT_APP_BACKEND_URL)
- [ ] CORS settings updated in backend
- [ ] Frontend can access backend API
- [ ] Test PCB inspection functionality
- [ ] Sample good/defective PCBs uploaded
- [ ] ML model can be trained
- [ ] Inspection history visible
- [ ] WebSocket connection working (real-time features)

---

## 🎉 Congratulations!

Your PCB Inspection System is now live on Render!

**Frontend URL:** `https://pcb-inspection-frontend.onrender.com`  
**Backend API:** `https://pcb-inspection-backend.onrender.com`

---

## 📝 Important Notes

### Camera Integration
- Camera features require physical hardware
- May not work in cloud deployments without specialized setup
- Consider disabling camera features for cloud deployment
- Use manual upload feature instead

### Dataset Management
- Store images in cloud storage (AWS S3, Cloudinary) for production
- Consider data persistence strategy
- Render's filesystem is ephemeral (resets on redeploy)

### Performance
- First inspection builds the ML model (slower)
- Subsequent inspections are faster
- Consider pre-training model during deployment

---

**Need Help?** Check the troubleshooting section or Render's documentation.

**Happy Deploying! 🚀**
