# ============================================================================
# RipCatch v2.0 - Quick Docker Setup Script (Windows)
# Automates Docker deployment on Windows
# ============================================================================

Write-Host "🌊 RipCatch v2.0 - Docker Quick Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check if Docker is installed
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "   Visit: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
    exit 1
}

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Ask user for deployment type
Write-Host ""
Write-Host "Select deployment type:"
Write-Host "1) CPU (recommended for testing)"
Write-Host "2) GPU (requires NVIDIA Docker runtime and WSL2)"
$choice = Read-Host "Enter choice [1-2]"

switch ($choice) {
    "1" {
        $DEPLOY_TYPE = "cpu"
        $IMAGE = "naga-narala/ripcatch:latest-cpu"
        $PORT = 7860
        $CONTAINER_NAME = "ripcatch"
    }
    "2" {
        $DEPLOY_TYPE = "gpu"
        $IMAGE = "naga-narala/ripcatch:latest-gpu"
        $PORT = 7860
        $CONTAINER_NAME = "ripcatch-gpu"
        
        Write-Host "⚠️  GPU support requires:" -ForegroundColor Yellow
        Write-Host "   - WSL2 with NVIDIA drivers" -ForegroundColor Yellow
        Write-Host "   - Docker Desktop with GPU support enabled" -ForegroundColor Yellow
    }
    default {
        Write-Host "❌ Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "📦 Selected: $DEPLOY_TYPE deployment" -ForegroundColor Cyan
Write-Host ""

# Ask to build or pull
Write-Host "Do you want to:"
Write-Host "1) Pull pre-built image from Docker Hub (faster)"
Write-Host "2) Build from source (requires model file)"
$build_choice = Read-Host "Enter choice [1-2]"

if ($build_choice -eq "2") {
    # Build from source
    Write-Host ""
    Write-Host "🔨 Building Docker image from source..." -ForegroundColor Cyan
    
    if ($DEPLOY_TYPE -eq "cpu") {
        docker build -t ripcatch:local-cpu -f Dockerfile .
        $IMAGE = "ripcatch:local-cpu"
    } else {
        docker build -t ripcatch:local-gpu -f Dockerfile.gpu .
        $IMAGE = "ripcatch:local-gpu"
    }
    
    Write-Host "✅ Build complete" -ForegroundColor Green
} else {
    # Pull from Docker Hub
    Write-Host ""
    Write-Host "📥 Pulling image from Docker Hub..." -ForegroundColor Cyan
    docker pull $IMAGE
    Write-Host "✅ Pull complete" -ForegroundColor Green
}

# Stop and remove existing container if exists
Write-Host ""
Write-Host "🧹 Cleaning up existing containers..." -ForegroundColor Cyan
docker stop $CONTAINER_NAME 2>$null
docker rm $CONTAINER_NAME 2>$null

# Create directories for volumes
New-Item -ItemType Directory -Force -Path "uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null

# Deploy
Write-Host ""
Write-Host "🚀 Starting RipCatch..." -ForegroundColor Cyan

if ($DEPLOY_TYPE -eq "cpu") {
    docker run -d `
        -p ${PORT}:7860 `
        --name $CONTAINER_NAME `
        --restart unless-stopped `
        -v "${PWD}/uploads:/app/uploads" `
        -v "${PWD}/outputs:/app/outputs" `
        $IMAGE
} else {
    docker run -d `
        -p ${PORT}:7860 `
        --name $CONTAINER_NAME `
        --gpus all `
        --restart unless-stopped `
        -v "${PWD}/uploads:/app/uploads" `
        -v "${PWD}/outputs:/app/outputs" `
        $IMAGE
}

# Wait for container to start
Write-Host ""
Write-Host "⏳ Waiting for container to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Check if container is running
$running = docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}"
if ($running -eq $CONTAINER_NAME) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "✅ RipCatch is now running!" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Access the app at: http://localhost:$PORT" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📊 Useful commands:" -ForegroundColor Yellow
    Write-Host "   View logs:    docker logs -f $CONTAINER_NAME"
    Write-Host "   Stop:         docker stop $CONTAINER_NAME"
    Write-Host "   Restart:      docker restart $CONTAINER_NAME"
    Write-Host "   Remove:       docker rm -f $CONTAINER_NAME"
    Write-Host ""
    Write-Host "📚 For more info, see: DOCKER_GUIDE.md" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Container failed to start. Check logs:" -ForegroundColor Red
    Write-Host "   docker logs $CONTAINER_NAME" -ForegroundColor Yellow
    exit 1
}
