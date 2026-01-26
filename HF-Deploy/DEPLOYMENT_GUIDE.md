# 🚀 RipCatch v2.0 - Hugging Face Deployment Guide

**Status**: Ready to Deploy ✅  
**Date**: January 27, 2026

---

## 📦 What's in this folder?

```
HF-Deploy/
├── .gitattributes          # Git LFS config for large model files
├── app.py                  # Gradio web interface
├── requirements.txt        # Python dependencies
├── README.md               # Hugging Face Space description
├── Demo.gif                # Demo animation (15.6 MB)
└── RipCatch-v2.0/
    └── Model/
        └── weights/
            └── best.pt     # YOLOv8m model (52 MB)
```

**Total size**: ~67.6 MB

---

## 🎯 Method 1: Web Interface Upload (EASIEST - 10 minutes)

### Step 1: Create Hugging Face Space

1. Go to: **https://huggingface.co/new-space**
2. Fill in the form:
   - **Owner**: Select your account
   - **Space name**: `ripcatch` (or `rip-current-detector`)
   - **License**: MIT
   - **Select the Space SDK**: **Gradio**
   - **Space hardware**: CPU (basic - free) or **GPU** (recommended for faster inference)
   - **Visibility**: Public

3. Click **"Create Space"**

### Step 2: Upload Files

After creating the Space, you'll see an empty repository. Upload files:

1. Click **"Files"** tab → **"Add file"** → **"Upload files"**

2. **Upload these files** (drag & drop from `HF-Deploy` folder):
   - `.gitattributes`
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `Demo.gif` (optional, but nice to have)

3. **Upload the model** (important!):
   - Create folder structure: Click **"Add file"** → **"Create a new file"**
   - Name it: `RipCatch-v2.0/Model/weights/.gitkeep`
   - Commit the file
   - Then upload `best.pt` to `RipCatch-v2.0/Model/weights/`

4. Commit changes with message: `Initial commit - RipCatch v2.0`

### Step 3: Wait for Build

- Hugging Face will automatically build your Space (~2-3 minutes)
- Watch the **"Logs"** tab for build progress
- Once complete, your app will be live! 🎉

### Step 4: Test Your Space

- Click on **"App"** tab
- Test with sample beach images
- Share the link: `https://huggingface.co/spaces/YOUR_USERNAME/ripcatch`

---

## 🎯 Method 2: Git Push (ADVANCED - 15 minutes)

This method requires Git LFS and Hugging Face CLI setup.

### Prerequisites

1. **Install Hugging Face CLI**:
   ```powershell
   pip install huggingface_hub[cli]
   ```

2. **Login to Hugging Face**:
   ```powershell
   huggingface-cli login
   ```
   - Get your token from: https://huggingface.co/settings/tokens
   - Paste token when prompted

3. **Install Git LFS** (for large model files):
   - Download from: https://git-lfs.github.com/
   - Or: `choco install git-lfs` (if using Chocolatey)
   - Run: `git lfs install`

### Deployment Steps

1. **Navigate to deployment folder**:
   ```powershell
   cd HF-Deploy
   ```

2. **Initialize Git repository**:
   ```powershell
   git init
   git lfs install
   git lfs track "*.pt"
   ```

3. **Add files and commit**:
   ```powershell
   git add .
   git commit -m "Initial commit: RipCatch v2.0"
   ```

4. **Add Hugging Face remote** (replace `YOUR_USERNAME`):
   ```powershell
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/ripcatch
   ```

5. **Push to Hugging Face**:
   ```powershell
   git push -u origin main
   ```

6. **Wait for build** and visit your Space!

---

## 🔧 Troubleshooting

### Issue: "Model file not found"
**Solution**: Make sure `best.pt` is uploaded to exactly this path:
```
RipCatch-v2.0/Model/weights/best.pt
```

### Issue: "Out of memory" on CPU
**Solution**: 
- Go to Space settings
- Upgrade to GPU hardware (T4 small - paid tier)
- Or reduce image size in app (already optimized to 640px)

### Issue: "Build failed - requirements error"
**Solution**: 
- Check `requirements.txt` is uploaded correctly
- Wait a few minutes and try again (HF servers sometimes busy)

### Issue: "Git LFS error" when pushing
**Solution**:
```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add LFS tracking"
git push
```

---

## 📊 Expected Performance

**On CPU (free tier)**:
- Image inference: ~3-5 seconds
- Video inference: ~30 FPS input → ~2 FPS output

**On GPU (T4 small - $0.60/hour)**:
- Image inference: ~0.5-1 second
- Video inference: ~10-15 FPS real-time

---

## 🎨 Customization Options

After deployment, you can customize:

1. **Title & Description**: Edit `README.md` in your Space
2. **Theme**: Edit `app.py` → change `theme="default"` to other Gradio themes
3. **Examples**: Add your own sample images in `app.py` → `examples` parameter
4. **Hardware**: Upgrade to GPU in Space settings for faster inference

---

## ✅ Post-Deployment Checklist

After successful deployment:

- [ ] Test image upload feature
- [ ] Test video upload feature
- [ ] Test webcam feature (if available)
- [ ] Verify model predictions are accurate
- [ ] Check confidence threshold slider works
- [ ] Share your Space on social media
- [ ] Add Space link to your GitHub README
- [ ] (Optional) Apply for Hugging Face GPU grant if doing research

---

## 🔗 Useful Links

- **Hugging Face Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Documentation**: https://gradio.app/docs/
- **Your GitHub Repo**: https://github.com/naga-narala/RipCatch
- **Git LFS**: https://git-lfs.github.com/

---

## 📧 Support

If you encounter issues:
1. Check Hugging Face Space logs (Logs tab)
2. Visit Hugging Face Discord: https://hf.co/join/discord
3. Open issue on GitHub: https://github.com/naga-narala/RipCatch/issues

---

**Good luck with your deployment! 🌊🚀**

*Once deployed, your Space URL will be: `https://huggingface.co/spaces/YOUR_USERNAME/ripcatch`*
