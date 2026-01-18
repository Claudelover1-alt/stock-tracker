# ✅ Quick Deployment Checklist

Follow these steps in order for fastest deployment:

## Step 1: Download Files (On Your MacBook)
- [ ] Open this conversation in Chrome on your MacBook
- [ ] Download all 9 files by clicking each download link
- [ ] Files should be in your Downloads folder

**Required files:**
1. configure_stocks.py
2. dashboard_multi.html
3. render.yaml
4. requirements.txt
5. server.py
6. stock_analyzer.py
7. stocks_config.json
8. .gitignore
9. DEPLOY.md (this guide)

## Step 2: Create GitHub Account (if needed)
- [ ] Go to https://github.com
- [ ] Click "Sign up"
- [ ] Use your email
- [ ] Choose "Free" plan
- [ ] Verify email

**Already have GitHub?** Just sign in and skip to Step 3.

## Step 3: Create Repository
- [ ] Click "+" in top-right on GitHub
- [ ] Click "New repository"
- [ ] Name: `stock-tracker`
- [ ] Select "Public"
- [ ] Check "Add a README file"
- [ ] Click "Create repository"

## Step 4: Upload Files
- [ ] Click "Add file" → "Upload files"
- [ ] Drag all 8 files from Downloads folder
- [ ] Click "Commit changes"

**Files to upload:**
- configure_stocks.py
- dashboard_multi.html
- render.yaml
- requirements.txt
- server.py
- stock_analyzer.py
- stocks_config.json
- .gitignore

## Step 5: Deploy to Render
- [ ] Go to https://render.com
- [ ] Click "Get Started"
- [ ] Sign up with GitHub
- [ ] Authorize Render
- [ ] Click "New +" → "Web Service"
- [ ] Click "Build and deploy from a Git repository"
- [ ] Connect your `stock-tracker` repository
- [ ] Configure:
  - Name: stock-tracker
  - Runtime: Python 3
  - Build: `pip install -r requirements.txt`
  - Start: `gunicorn server:app`
  - Instance Type: **Free**
- [ ] Click "Create Web Service"
- [ ] Wait 3-5 minutes for deployment

## Step 6: Access Your Dashboard
- [ ] Copy the URL from Render (looks like `https://stock-tracker-xxxx.onrender.com`)
- [ ] Open in browser to test
- [ ] Open on iPhone Safari
- [ ] Tap Share → Add to Home Screen
- [ ] Name it "Stock Tracker"
- [ ] Tap "Add"

## 🎉 Done!

Your dashboard is now live 24/7 at your Render URL.

## 🔧 Quick Edits

To change stocks or targets:
1. Go to GitHub repository
2. Click `stocks_config.json`
3. Click pencil icon to edit
4. Make changes
5. Click "Commit changes"
6. Wait 2-3 minutes for Render to auto-deploy

## ⏱️ Total Time
- First time setup: 10 minutes
- Already have accounts: 3 minutes
- Future edits: 30 seconds

## 📱 Pro Tip
Bookmark your Render URL in Chrome on MacBook for quick access to check stocks on your computer too!
