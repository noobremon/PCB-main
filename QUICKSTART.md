# 🚀 Quick Start: Deploy to Render in 5 Steps

## Prerequisites
- [ ] Render account created (https://render.com)
- [ ] MongoDB Atlas account created (https://www.mongodb.com/cloud/atlas)
- [ ] Code pushed to GitHub/GitLab

**Note:** No Docker needed! Render uses native Python/Node.js buildpacks.

---

## Step 1: Setup MongoDB (5 minutes)

1. Go to https://cloud.mongodb.com
2. Create free cluster (M0)
3. Create database user
4. Whitelist IP: `0.0.0.0/0`
5. Copy connection string

---

## Step 2: Deploy Backend as Web Service (5 minutes)

1. Go to https://dashboard.render.com
2. Click **New +** → **Web Service**
3. Connect repository: `PCB-main`
4. Configure:
   - **Name:** `pcb-inspection-backend`
   - **Root Directory:** `backend` ⚠️ **Must specify!**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Click **Advanced** → Add Environment Variables:
   ```
   MONGO_URL = your_mongodb_connection_string_from_step_1
   DB_NAME = pcb_inspection
   LOG_LEVEL = INFO
   ```
6. Click **Create Web Service**
7. Wait 5-10 minutes, then **copy backend URL**
   - Example: `https://pcb-inspection-backend.onrender.com`

---

## Step 3: Deploy Frontend as Static Site (5 minutes)

1. In Render Dashboard, click **New +** → **Static Site**
2. Select same repository: `PCB-main`
3. Configure:
   - **Name:** `pcb-inspection-frontend`
   - **Root Directory:** `frontend` ⚠️ **Must specify!**
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `build`
4. Click **Advanced** → Add Environment Variable:
   ```
   REACT_APP_BACKEND_URL = https://pcb-inspection-backend.onrender.com
   ```
   (Use your actual backend URL from Step 2, no trailing slash)
5. Click **Create Static Site**
6. Wait 3-5 minutes → Frontend URL is your live app! 🎉

---

## Step 4: Update CORS (2 minutes)

1. Edit [backend/server.py](backend/server.py) line ~80
2. Change:
   ```python
   allow_origins=["*"]
   ```
   To:
   ```python
   allow_origins=["https://your-frontend-url.onrender.com", "http://localhost:3000"]
   ```
3. Commit and push → Auto-deploys

---

## Step 5: Test (1 minute)

1. Visit your frontend URL
2. Upload a PCB image
3. See inspection results!

---

## ✅ Done!

Your PCB inspection system is live!

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed documentation.

---

## 🐛 Quick Troubleshooting

**Backend won't start?**
- Check MongoDB connection string
- Verify environment variables

**Frontend can't connect?**
- Check REACT_APP_BACKEND_URL is correct
- Update CORS settings

**First request is slow?**
- Normal on free tier (cold start)
- Upgrade to paid plan for 24/7 uptime

---

## 💡 Tips

- Free tier services sleep after 15min inactivity
- First request after sleep takes 30-60 seconds
- Upgrade to **Starter** ($7/mo) for always-on backend

---

**Need help?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for full guide.
