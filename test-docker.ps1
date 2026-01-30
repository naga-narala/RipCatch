# ============================================================================
# RipCatch v2.0 - Local Docker Test Script
# Tests Docker build and deployment locally
# ============================================================================

Write-Host "🌊 RipCatch v2.0 - Local Docker Test" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "🔍 Checking Docker status..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running!" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "📦 Which version would you like to test?" -ForegroundColor Cyan
Write-Host "1) CPU version (recommended for testing)"
Write-Host "2) GPU version (requires NVIDIA GPU + drivers)"
Write-Host "3) Both"
$choice = Read-Host "Enter choice [1-3]"

$BUILD_CPU = $false
$BUILD_GPU = $false

switch ($choice) {
    "1" { $BUILD_CPU = $true }
    "2" { $BUILD_GPU = $true }
    "3" { $BUILD_CPU = $true; $BUILD_GPU = $true }
    default {
        Write-Host "❌ Invalid choice" -ForegroundColor Red
        exit 1
    }
}

# Test CPU Build
if ($BUILD_CPU) {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "🔨 Building CPU Docker Image" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will take 3-5 minutes (downloading dependencies)..." -ForegroundColor Yellow
    Write-Host ""
    
    $startTime = Get-Date
    docker build -t ripcatch:test-cpu -f Dockerfile .
    
    if ($LASTEXITCODE -eq 0) {
        $buildTime = (Get-Date) - $startTime
        Write-Host ""
        Write-Host "✅ CPU build successful! (took $($buildTime.TotalMinutes.ToString('0.0')) minutes)" -ForegroundColor Green
        
        # Check image size
        $imageInfo = docker images ripcatch:test-cpu --format "{{.Size}}"
        Write-Host "📦 Image size: $imageInfo" -ForegroundColor Cyan
        
        # Run container
        Write-Host ""
        Write-Host "🚀 Starting CPU container..." -ForegroundColor Cyan
        
        # Stop any existing container
        docker stop ripcatch-test-cpu 2>$null
        docker rm ripcatch-test-cpu 2>$null
        
        docker run -d `
            -p 7860:7860 `
            --name ripcatch-test-cpu `
            ripcatch:test-cpu
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Container started successfully!" -ForegroundColor Green
            Write-Host ""
            Write-Host "⏳ Waiting for app to start (30 seconds)..." -ForegroundColor Yellow
            Start-Sleep -Seconds 30
            
            Write-Host ""
            Write-Host "=====================================" -ForegroundColor Green
            Write-Host "✅ CPU TEST SUCCESSFUL!" -ForegroundColor Green
            Write-Host "=====================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "🌐 Access the app at: http://localhost:7860" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "📊 Container Info:" -ForegroundColor Yellow
            docker ps --filter "name=ripcatch-test-cpu" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            Write-Host ""
            Write-Host "📝 View logs: docker logs -f ripcatch-test-cpu" -ForegroundColor Yellow
            Write-Host "🛑 Stop:      docker stop ripcatch-test-cpu" -ForegroundColor Yellow
            Write-Host "🗑️  Remove:    docker rm -f ripcatch-test-cpu" -ForegroundColor Yellow
            Write-Host ""
        } else {
            Write-Host "❌ Failed to start container" -ForegroundColor Red
            Write-Host "Check logs: docker logs ripcatch-test-cpu" -ForegroundColor Yellow
        }
    } else {
        Write-Host ""
        Write-Host "❌ CPU build failed!" -ForegroundColor Red
        Write-Host "Check the error messages above." -ForegroundColor Yellow
    }
}

# Test GPU Build
if ($BUILD_GPU) {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "🔨 Building GPU Docker Image" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check for NVIDIA GPU
    try {
        nvidia-smi | Out-Null
        Write-Host "✅ NVIDIA GPU detected" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  NVIDIA GPU not detected or drivers not installed" -ForegroundColor Yellow
        Write-Host "GPU container may not work properly." -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "This will take 5-10 minutes (larger base image)..." -ForegroundColor Yellow
    Write-Host ""
    
    $startTime = Get-Date
    docker build -t ripcatch:test-gpu -f Dockerfile.gpu .
    
    if ($LASTEXITCODE -eq 0) {
        $buildTime = (Get-Date) - $startTime
        Write-Host ""
        Write-Host "✅ GPU build successful! (took $($buildTime.TotalMinutes.ToString('0.0')) minutes)" -ForegroundColor Green
        
        # Check image size
        $imageInfo = docker images ripcatch:test-gpu --format "{{.Size}}"
        Write-Host "📦 Image size: $imageInfo" -ForegroundColor Cyan
        
        # Run container
        Write-Host ""
        Write-Host "🚀 Starting GPU container..." -ForegroundColor Cyan
        
        # Stop any existing container
        docker stop ripcatch-test-gpu 2>$null
        docker rm ripcatch-test-gpu 2>$null
        
        docker run -d `
            -p 7861:7860 `
            --name ripcatch-test-gpu `
            --gpus all `
            ripcatch:test-gpu
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Container started successfully!" -ForegroundColor Green
            Write-Host ""
            Write-Host "⏳ Waiting for app to start (30 seconds)..." -ForegroundColor Yellow
            Start-Sleep -Seconds 30
            
            Write-Host ""
            Write-Host "=====================================" -ForegroundColor Green
            Write-Host "✅ GPU TEST SUCCESSFUL!" -ForegroundColor Green
            Write-Host "=====================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "🌐 Access the app at: http://localhost:7861" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "📊 Container Info:" -ForegroundColor Yellow
            docker ps --filter "name=ripcatch-test-gpu" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            Write-Host ""
            Write-Host "📝 View logs: docker logs -f ripcatch-test-gpu" -ForegroundColor Yellow
            Write-Host "🛑 Stop:      docker stop ripcatch-test-gpu" -ForegroundColor Yellow
            Write-Host "🗑️  Remove:    docker rm -f ripcatch-test-gpu" -ForegroundColor Yellow
            Write-Host ""
        } else {
            Write-Host "❌ Failed to start container" -ForegroundColor Red
            Write-Host "This might be because:" -ForegroundColor Yellow
            Write-Host "- NVIDIA Docker runtime not installed" -ForegroundColor Yellow
            Write-Host "- GPU drivers not available" -ForegroundColor Yellow
            Write-Host "Check logs: docker logs ripcatch-test-gpu" -ForegroundColor Yellow
        }
    } else {
        Write-Host ""
        Write-Host "❌ GPU build failed!" -ForegroundColor Red
        Write-Host "Check the error messages above." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "🏁 Test Complete!" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor Yellow
docker images ripcatch:test-* --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
Write-Host ""

if ($BUILD_CPU) {
    Write-Host "🌐 CPU App: http://localhost:7860" -ForegroundColor Green
}
if ($BUILD_GPU) {
    Write-Host "🌐 GPU App: http://localhost:7861" -ForegroundColor Green
}

Write-Host ""
Write-Host "🧹 To clean up after testing:" -ForegroundColor Yellow
Write-Host "   docker stop ripcatch-test-cpu ripcatch-test-gpu" -ForegroundColor White
Write-Host "   docker rm ripcatch-test-cpu ripcatch-test-gpu" -ForegroundColor White
Write-Host "   docker rmi ripcatch:test-cpu ripcatch:test-gpu" -ForegroundColor White
Write-Host ""
