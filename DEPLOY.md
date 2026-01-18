# 🚀 Deploy to Render (Free Cloud Hosting)

This guide will help you deploy your Stock Tracker to Render's free tier, giving you 24/7 access from anywhere without keeping your computer running!

## ⏱️ Time Required: 5-10 minutes

## 📋 What You'll Need
- GitHub account (free - we'll create one if you don't have it)
- Render account (free - no credit card required)
- The files you downloaded

## 🎯 Step-by-Step Deployment

### Part 1: Set Up GitHub (2 minutes)

#### If you already have a GitHub account:
1. Go to https://github.com and sign in
2. Skip to Part 2

#### If you need to create a GitHub account:
1. Go to https://github.com
2. Click **"Sign up"**
3. Enter your email address
4. Create a password
5. Choose a username
6. Verify your account (check email)
7. Choose "Free" plan

### Part 2: Upload Files to GitHub (3 minutes)

1. **Go to GitHub and create a new repository:**
   - Click the **"+"** in top-right corner
   - Select **"New repository"**

2. **Configure your repository:**
   - Repository name: `stock-tracker` (or whatever you prefer)
   - Description: "Multi-stock analysis dashboard"
   - Select **"Public"** (required for free Render deployment)
   - ✅ Check **"Add a README file"**
   - Click **"Create repository"**

3. **Upload your files:**
   - Click **"Add file"** → **"Upload files"**
   - Drag and drop ALL these files into the upload area:
     - `configure_stocks.py`
     - `dashboard_multi.html`
     - `render.yaml`
     - `requirements.txt`
     - `server.py`
     - `stock_analyzer.py`
     - `stocks_config.json`
     - `.gitignore`
   - Scroll down and click **"Commit changes"**

### Part 3: Deploy to Render (2 minutes)

1. **Create Render account:**
   - Go to https://render.com
   - Click **"Get Started"**
   - Sign up with your **GitHub account** (easiest)
   - Authorize Render to access GitHub

2. **Create new Web Service:**
   - Click **"New +"** in top-right
   - Select **"Web Service"**
   - Click **"Build and deploy from a Git repository"**
   - Click **"Next"**

3. **Connect your repository:**
   - Find your `stock-tracker` repository
   - Click **"Connect"**

4. **Configure the service:**
   - **Name:** `stock-tracker` (or whatever you want)
   - **Region:** Choose closest to you
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn server:app`
   - **Instance Type:** Select **"Free"**

5. **Deploy:**
   - Click **"Create Web Service"**
   - Wait 3-5 minutes for deployment (you'll see build logs)
   - When you see "Your service is live 🎉" it's ready!

### Part 4: Access Your Dashboard (1 minute)

1. **Get your URL:**
   - At the top of your Render dashboard, you'll see a URL like:
   - `https://stock-tracker-xxxx.onrender.com`
   - Click it to open your dashboard!

2. **Add to iPhone:**
   - Open the URL in Safari on your iPhone
   - Tap the **Share** button
   - Tap **"Add to Home Screen"**
   - Name it "Stock Tracker"
   - Tap **"Add"**

## ✅ You're Done!

Your stock tracker is now:
- ✅ Running 24/7 in the cloud
- ✅ Accessible from anywhere
- ✅ No computer needed
- ✅ Completely free
- ✅ Automatically updates every second

## 🎨 Customizing Your Stocks

To change which stocks you're tracking:

1. **Edit `stocks_config.json` on GitHub:**
   - Go to your repository on GitHub
   - Click on `stocks_config.json`
   - Click the **pencil icon** (Edit)
   - Modify the stocks and target prices
   - Click **"Commit changes"**

2. **Render will auto-deploy:**
   - Render detects the change automatically
   - Rebuilds and redeploys (takes 2-3 minutes)
   - Your dashboard updates with new stocks!

### Example: Change RCAT target from $20 to $25
```json
{
  "stocks": [
    {
      "ticker": "RCAT",
      "target_price": 25.00,  ← Change this number
      "enabled": true
    }
  ]
}
```

## 🔄 Managing Your Stocks

### Add a New Stock
1. Edit `stocks_config.json` on GitHub
2. Add a new stock entry:
```json
{
  "ticker": "NVDA",
  "target_price": 200.00,
  "enabled": true
}
```
3. Commit changes
4. Wait for auto-deploy

### Remove a Stock
1. Edit `stocks_config.json` on GitHub
2. Either delete the entire stock entry OR set `"enabled": false`
3. Commit changes

### Change Target Prices
1. Edit `stocks_config.json` on GitHub
2. Update the `target_price` value
3. Commit changes

## ⚙️ Advanced Settings

### Change Update Speed
Edit `stocks_config.json`:
```json
{
  "update_interval_seconds": 5  ← Change from 1 to 5 for every 5 seconds
}
```

### View Logs
1. Go to your Render dashboard
2. Click on your service
3. Click "Logs" tab
4. See real-time analysis updates

### Restart Service
1. Go to Render dashboard
2. Click "Manual Deploy" → "Clear build cache & deploy"

## 🆓 Free Tier Limitations

Render's free tier includes:
- ✅ 750 hours/month (more than enough - that's 24/7)
- ✅ Automatic HTTPS
- ✅ Auto-deploys from GitHub
- ⚠️ Spins down after 15 minutes of inactivity
- ⚠️ Cold start takes 30-60 seconds when accessing after inactivity

**What this means:** If you don't access the dashboard for 15 minutes, it goes to sleep. The first access after sleep takes 30-60 seconds to wake up, then it's instant again.

## 🐛 Troubleshooting

### "Build failed"
- Check that all files were uploaded to GitHub
- Verify `stocks_config.json` has valid JSON (no syntax errors)
- Check build logs in Render for specific error

### "Service unavailable"
- Wait 30-60 seconds (cold start from sleep)
- Check Render dashboard for service status
- View logs for errors

### Stocks not updating
- Verify stock tickers are valid US symbols
- Check `"enabled": true` in config
- View logs to see if analysis is running

### Can't access URL
- Make sure service shows "Live" status in Render
- Try the URL in incognito/private mode
- Clear browser cache

## 💰 Upgrade Options (Optional)

If you want to eliminate cold starts:
- Render Starter plan: $7/month
- Keeps service always running (no 15-minute sleep)
- Faster performance
- Priority support

**For most users, the free tier is perfect!**

## 🎉 Success!

You now have a professional stock tracking dashboard running in the cloud, accessible from anywhere on any device, completely free!

Share your URL with friends, access it from any device, and never worry about keeping your computer running again.

## 📧 Questions?

Check the main README.md for detailed documentation on:
- How the analysis works
- Understanding the metrics
- Customization options
- API endpoints
