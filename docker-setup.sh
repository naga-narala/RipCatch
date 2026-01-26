#!/bin/bash
# ============================================================================
# RipCatch v2.0 - Quick Docker Setup Script
# Automates Docker deployment
# ============================================================================

set -e

echo "🌊 RipCatch v2.0 - Docker Quick Setup"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker is installed"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "⚠️  Docker Compose not found. Will use docker run instead."
    USE_COMPOSE=false
else
    echo "✅ Docker Compose is available"
    USE_COMPOSE=true
fi

# Ask user for deployment type
echo ""
echo "Select deployment type:"
echo "1) CPU (recommended for testing)"
echo "2) GPU (requires NVIDIA Docker runtime)"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        DEPLOY_TYPE="cpu"
        IMAGE="naga-narala/ripcatch:latest-cpu"
        PORT=7860
        ;;
    2)
        DEPLOY_TYPE="gpu"
        IMAGE="naga-narala/ripcatch:latest-gpu"
        PORT=7860
        
        # Check for NVIDIA Docker runtime
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "❌ NVIDIA Docker runtime not available"
            echo "   Install from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
        fi
        echo "✅ NVIDIA Docker runtime detected"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "📦 Selected: $DEPLOY_TYPE deployment"
echo ""

# Ask to build or pull
echo "Do you want to:"
echo "1) Pull pre-built image from Docker Hub (faster)"
echo "2) Build from source (requires model file)"
read -p "Enter choice [1-2]: " build_choice

if [ "$build_choice" = "2" ]; then
    # Build from source
    echo ""
    echo "🔨 Building Docker image from source..."
    
    if [ "$DEPLOY_TYPE" = "cpu" ]; then
        docker build -t ripcatch:local-cpu -f Dockerfile .
        IMAGE="ripcatch:local-cpu"
    else
        docker build -t ripcatch:local-gpu -f Dockerfile.gpu .
        IMAGE="ripcatch:local-gpu"
    fi
    
    echo "✅ Build complete"
else
    # Pull from Docker Hub
    echo ""
    echo "📥 Pulling image from Docker Hub..."
    docker pull $IMAGE
    echo "✅ Pull complete"
fi

# Deploy
echo ""
echo "🚀 Starting RipCatch..."

if [ "$USE_COMPOSE" = true ]; then
    # Use Docker Compose
    if [ "$DEPLOY_TYPE" = "cpu" ]; then
        docker-compose up -d ripcatch-cpu
    else
        docker-compose --profile gpu up -d ripcatch-gpu
    fi
else
    # Use docker run
    if [ "$DEPLOY_TYPE" = "cpu" ]; then
        docker run -d \
            -p $PORT:7860 \
            --name ripcatch \
            --restart unless-stopped \
            -v $(pwd)/uploads:/app/uploads \
            -v $(pwd)/outputs:/app/outputs \
            $IMAGE
    else
        docker run -d \
            -p $PORT:7860 \
            --name ripcatch-gpu \
            --gpus all \
            --restart unless-stopped \
            -v $(pwd)/uploads:/app/uploads \
            -v $(pwd)/outputs:/app/outputs \
            $IMAGE
    fi
fi

echo ""
echo "======================================"
echo "✅ RipCatch is now running!"
echo "======================================"
echo ""
echo "🌐 Access the app at: http://localhost:$PORT"
echo ""
echo "📊 Useful commands:"
echo "   View logs:    docker logs -f ripcatch"
echo "   Stop:         docker stop ripcatch"
echo "   Restart:      docker restart ripcatch"
echo "   Remove:       docker rm -f ripcatch"
echo ""
echo "📚 For more info, see: DOCKER_GUIDE.md"
echo ""
