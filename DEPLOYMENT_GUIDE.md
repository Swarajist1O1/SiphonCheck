# Diabetes Predictor - Deployment Guide

## 🚀 Deploy to Heroku (Recommended)

### Prerequisites
1. Create a Heroku account at https://heroku.com
2. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

### Step-by-Step Deployment

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-diabetes-predictor-app
   # Replace 'your-diabetes-predictor-app' with your desired app name
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set DEBUG=False
   ```

4. **Deploy to Heroku**
   ```bash
   git add .
   git commit -m "Deploy diabetes predictor to Heroku"
   git push heroku main
   # Or: git push heroku master (if your branch is master)
   ```

5. **Open Your App**
   ```bash
   heroku open
   ```

### Your app will be available at:
`https://your-diabetes-predictor-app.herokuapp.com`

---

## 🌐 Alternative: Deploy to Render (Free)

### Prerequisites
1. Create account at https://render.com
2. Connect your GitHub repository

### Steps:
1. Go to Render Dashboard
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: Python 3

---

## 🔧 Alternative: Deploy to Railway

### Prerequisites
1. Create account at https://railway.app
2. Install Railway CLI

### Steps:
1. **Login and Initialize**
   ```bash
   railway login
   railway init
   ```

2. **Deploy**
   ```bash
   railway up
   ```

---

## 📋 Pre-Deployment Checklist

✅ **Files Ready:**
- [x] `app.py` - Main Flask application
- [x] `requirements.txt` - Dependencies
- [x] `Procfile` - Heroku process file
- [x] `runtime.txt` - Python version
- [x] `static/work.jpg` - Background image
- [x] `templates/` - HTML templates
- [x] Model files (`.pkl`)

✅ **Configuration:**
- [x] App configured for production (`host='0.0.0.0'`)
- [x] Port configuration from environment variable
- [x] Debug mode controlled by environment variable

---

## 🎯 Quick Deploy Commands

### For Heroku:
```bash
# 1. Create app
heroku create your-app-name

# 2. Deploy
git push heroku main

# 3. Open
heroku open
```

### For Render:
1. Visit https://render.com
2. Connect GitHub repo
3. Deploy automatically

---

## 🔍 Troubleshooting

### Common Issues:
1. **Slug size too large**: Remove unnecessary files
2. **Build failed**: Check requirements.txt
3. **App crashed**: Check Heroku logs: `heroku logs --tail`

### Environment Variables:
- `DEBUG=False` (for production)
- `PORT` (automatically set by platform)

---

## 📱 Access Your App

Once deployed, your Diabetes Predictor will be available at:
- **Heroku**: `https://your-app-name.herokuapp.com`
- **Render**: `https://your-app-name.onrender.com`
- **Railway**: `https://your-app-name.railway.app`

## ✨ Features Available:
- ✅ Transparent glass-morphism design
- ✅ Background image (work.jpg)
- ✅ Preset form values
- ✅ AI-powered diabetes risk prediction
- ✅ Clean, minimal interface
- ✅ Mobile responsive design
