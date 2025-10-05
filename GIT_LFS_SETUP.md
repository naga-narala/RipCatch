# 📦 Git LFS Setup for RipCatch

## Why Git LFS?

Your video files are too large for standard GitHub:
- `video_test_1_output.mp4`: 677 MB ❌
- `video_test_2_output.mp4`: 347 MB ❌

GitHub limit: 100 MB per file

Git LFS (Large File Storage) allows you to store large files up to **2 GB per file** and **5 GB total** on the free plan.

---

## 🚀 Quick Setup

### Step 1: Install Git LFS

**Windows:**
```powershell
# Using winget
winget install git-lfs

# Or download installer from
# https://git-lfs.github.com/
```

**macOS:**
```bash
brew install git-lfs
```

**Linux:**
```bash
sudo apt install git-lfs
```

### Step 2: Initialize Git LFS in Your Repo

```bash
# Navigate to your project
cd A:\5_projects\rip_current_project

# Initialize Git LFS
git lfs install

# Verify .gitattributes is present (already created)
cat .gitattributes
```

### Step 3: Track Your Large Files

The `.gitattributes` file is already configured to track:
- `*.mp4` files
- `RipCatch-v2.0/Results/*.mp4`
- Large model weights (*.pt, *.pth, *.onnx)
- `inference_results.png`

### Step 4: Add and Commit Files

```bash
# Add all files (LFS will handle large files automatically)
git add .gitignore
git add .gitattributes
git add README.md
git add Demo.gif
git add Demo.mp4

# Add large files (will be tracked by LFS)
git add RipCatch-v2.0/Results/
git add Testing/videos/

# Commit changes
git commit -m "Add demo video, testing videos, and v2.0 results with Git LFS"

# Push to GitHub
git push origin main
```

### Step 5: Verify LFS is Working

```bash
# Check which files are tracked by LFS
git lfs ls-files

# Should show your large MP4 files
```

---

## 📊 Git LFS Limits

### GitHub Free Plan
- **File size limit**: 2 GB per file
- **Storage limit**: 1 GB free storage
- **Bandwidth**: 1 GB/month free bandwidth

### If You Exceed Limits
- **Upgrade to GitHub Pro**: $4/month (50 GB storage, 50 GB bandwidth)
- **Buy data packs**: $5/month per 50 GB storage + 50 GB bandwidth
- **Or use external hosting** (see Option 2 below)

---

## 🎯 Alternative: External Hosting (Option 2)

If you don't want to use Git LFS:

### Host Videos Externally

1. **Google Drive / Dropbox**
   ```
   Upload videos → Get shareable link → Update README
   ```

2. **YouTube (Unlisted)**
   ```
   Upload as unlisted video → Embed in README
   ```

3. **GitHub Releases**
   ```
   Create a Release → Attach videos as release assets
   (Up to 2 GB per asset)
   ```

### Update README to Link External Videos

Instead of including videos directly, add download links:

```markdown
## 🎬 Demo Videos

**Download Full Demo Videos:**
- [video_test_1_output.mp4 (677 MB)](https://drive.google.com/file/d/YOUR_FILE_ID)
- [video_test_2_output.mp4 (347 MB)](https://drive.google.com/file/d/YOUR_FILE_ID)

**Or watch on YouTube:**
- [Test Video 1 - Beach Surveillance](https://youtube.com/watch?v=YOUR_VIDEO_ID)
```

### Keep Only Small Files in Repo

```bash
# Remove large videos from staging
git reset RipCatch-v2.0/Results/*.mp4

# Update .gitignore to exclude them
echo "RipCatch-v2.0/Results/*.mp4" >> .gitignore

# Keep only small files
git add RipCatch-v2.0/Results/evaluation_results.json
git add RipCatch-v2.0/Results/results.csv
# Compress inference_results.png before adding
```

---

## 🔧 Compress Large Files (Option 3)

### Compress Videos

Reduce video size using FFmpeg:

```bash
# Compress video_test_1_output.mp4 (677 MB → ~100 MB)
ffmpeg -i RipCatch-v2.0/Results/video_test_1_output.mp4 \
       -vcodec libx264 -crf 28 -preset fast \
       RipCatch-v2.0/Results/video_test_1_compressed.mp4

# Compress video_test_2_output.mp4 (347 MB → ~50 MB)
ffmpeg -i RipCatch-v2.0/Results/video_test_2_output.mp4 \
       -vcodec libx264 -crf 28 -preset fast \
       RipCatch-v2.0/Results/video_test_2_compressed.mp4
```

### Compress PNG

Reduce PNG size using ImageMagick or online tools:

```bash
# Using ImageMagick
magick RipCatch-v2.0/Results/inference_results.png \
       -quality 85 -resize 50% \
       RipCatch-v2.0/Results/inference_results_compressed.png

# Or use online: https://tinypng.com/
```

---

## 📝 Recommended Approach

**For Your Project:**

1. ✅ **Use Git LFS** for:
   - `Demo.mp4` (13 MB - OK but good to track)
   - Result videos (677 MB + 347 MB)
   - Large model weights if you add them later

2. ✅ **Keep in repo** (regular Git):
   - `Demo.gif` (15.6 MB - displays inline)
   - `inference_results.png` (62.9 MB - can compress to ~10 MB)
   - Testing images (already tracked)
   - JSON/CSV results

3. ✅ **Compress if needed**:
   - Compress `inference_results.png` to under 10 MB
   - Optionally compress result videos to under 100 MB each

---

## 🚀 Quick Start Commands

```bash
# Install and setup Git LFS
git lfs install

# Files are already configured in .gitattributes

# Add everything
git add .

# Commit
git commit -m "Add demo, testing videos, and results with Git LFS"

# Push (will upload large files to LFS)
git push origin main
```

---

## ❓ Troubleshooting

**Issue**: "git lfs not found"
```bash
# Reinstall Git LFS
winget install git-lfs
git lfs install --force
```

**Issue**: "Exceeded bandwidth quota"
```bash
# Wait for next month, or upgrade to GitHub Pro
# Or move files to external hosting
```

**Issue**: "File too large for Git LFS"
```bash
# Maximum file size is 2 GB per file
# Compress or split large files
```

---

## 📚 More Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub LFS Pricing](https://docs.github.com/en/billing/managing-billing-for-git-large-file-storage)
- [FFmpeg Video Compression](https://trac.ffmpeg.org/wiki/Encode/H.264)

---

**Questions?** Check the [Git LFS FAQ](https://github.com/git-lfs/git-lfs/wiki/FAQ) or create an issue.

