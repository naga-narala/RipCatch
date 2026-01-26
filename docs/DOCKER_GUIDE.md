# 🐳 RipCatch v2.0 - Docker Deployment Guide

Complete guide for deploying RipCatch v2.0 using Docker and Kubernetes.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- (For GPU) NVIDIA Docker Runtime
- 4GB+ RAM available
- 10GB+ disk space

### Run with Docker (CPU)

```bash
# Pull pre-built image
docker pull naga-narala/ripcatch:latest-cpu

# Run container
docker run -d -p 7860:7860 --name ripcatch naga-narala/ripcatch:latest-cpu

# Access at http://localhost:7860
```

### Run with Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/naga-narala/RipCatch.git
cd RipCatch

# Start CPU service
docker-compose up -d ripcatch-cpu

# Or start GPU service
docker-compose --profile gpu up -d ripcatch-gpu
```

---

## 🐳 Docker Deployment

### Option 1: Pre-built Images (Docker Hub)

**CPU Version:**
```bash
docker pull naga-narala/ripcatch:latest-cpu
docker run -d \
  -p 7860:7860 \
  --name ripcatch \
  --restart unless-stopped \
  naga-narala/ripcatch:latest-cpu
```

**GPU Version:**
```bash
docker pull naga-narala/ripcatch:latest-gpu
docker run -d \
  -p 7860:7860 \
  --name ripcatch-gpu \
  --gpus all \
  --restart unless-stopped \
  naga-narala/ripcatch:latest-gpu
```

### Option 2: Build from Source

**Build CPU Image:**
```bash
docker build -t ripcatch:local-cpu -f Dockerfile .
```

**Build GPU Image:**
```bash
docker build -t ripcatch:local-gpu -f Dockerfile.gpu .
```

### Option 3: Docker Compose

**Configuration:**
```yaml
# docker-compose.yml is already configured
# Edit environment variables if needed
```

**Start Services:**
```bash
# CPU service (port 7860)
docker-compose up -d ripcatch-cpu

# GPU service (port 7861)
docker-compose --profile gpu up -d ripcatch-gpu

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Persistent Storage

Mount volumes for uploads and outputs:

```bash
docker run -d \
  -p 7860:7860 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  --name ripcatch \
  naga-narala/ripcatch:latest-cpu
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- (For GPU) NVIDIA device plugin installed

### Deploy CPU Version

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -l app=ripcatch
kubectl get svc ripcatch-service

# Get external IP
kubectl get svc ripcatch-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### Deploy GPU Version

```bash
# Ensure GPU nodes are labeled
kubectl label nodes <node-name> accelerator=nvidia-gpu

# Apply GPU deployment
kubectl apply -f k8s/deployment-gpu.yaml

# Check status
kubectl get pods -l app=ripcatch-gpu
```

### Configure Ingress (Optional)

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Update domain in k8s/ingress.yaml
# Then apply
kubectl apply -f k8s/ingress.yaml
```

### Scaling

```bash
# Scale up
kubectl scale deployment ripcatch-deployment --replicas=5

# Autoscaling
kubectl autoscale deployment ripcatch-deployment \
  --cpu-percent=70 \
  --min=2 \
  --max=10
```

### Monitoring

```bash
# View logs
kubectl logs -f deployment/ripcatch-deployment

# Describe pod
kubectl describe pod <pod-name>

# Execute into container
kubectl exec -it <pod-name> -- bash
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Setup

The project includes automated Docker builds via GitHub Actions.

**Setup Secrets:**

1. Go to GitHub repo → Settings → Secrets and variables → Actions
2. Add secrets:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub token

**Workflow Triggers:**

- Push to `main` or `develop` → Build and push images
- Create tag `v*` → Build versioned release
- Pull Request → Build only (no push)

**Manual Trigger:**

```bash
# Go to Actions tab → Docker Image CI/CD → Run workflow
```

### Docker Hub Auto-Build

Images are automatically built and pushed to:
- `naga-narala/ripcatch:latest-cpu`
- `naga-narala/ripcatch:latest-gpu`
- `naga-narala/ripcatch:v2.0-cpu` (on version tags)
- `naga-narala/ripcatch:v2.0-gpu` (on version tags)

---

## 🛠️ Advanced Configuration

### Environment Variables

```bash
# Gradio server settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# Model settings (in app.py)
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45

# GPU settings
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
```

### Custom Build Arguments

```dockerfile
# Build with specific base image
docker build \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg CUDA_VERSION=11.8.0 \
  -t ripcatch:custom \
  -f Dockerfile.gpu .
```

### Multi-Architecture Builds

```bash
# Build for ARM64 (e.g., Apple Silicon, ARM servers)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t naga-narala/ripcatch:latest-cpu \
  --push \
  -f Dockerfile .
```

---

## 🔍 Troubleshooting

### Issue: Container exits immediately

**Solution:**
```bash
# Check logs
docker logs ripcatch

# Run interactively
docker run -it --rm naga-narala/ripcatch:latest-cpu bash
```

### Issue: GPU not detected

**Solution:**
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify Docker can access GPU
docker info | grep -i runtime
```

### Issue: Out of memory

**Solution:**
```bash
# Limit memory
docker run -d \
  -p 7860:7860 \
  --memory="4g" \
  --memory-swap="6g" \
  naga-narala/ripcatch:latest-cpu
```

### Issue: Port already in use

**Solution:**
```bash
# Use different port
docker run -d -p 8080:7860 naga-narala/ripcatch:latest-cpu

# Or stop conflicting container
docker ps
docker stop <container-id>
```

### Issue: Build fails - model file not found

**Solution:**
```bash
# Ensure model file exists
ls -lh RipCatch-v2.0/Model/weights/best.pt

# Check .dockerignore
cat .dockerignore | grep -v "!.*best.pt"
```

### Issue: Kubernetes pod crash loop

**Solution:**
```bash
# Check pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name> --previous

# Check resource limits
kubectl top pods
```

---

## 📊 Performance Benchmarks

### CPU Performance (Docker)
- Image inference: ~3-5 seconds
- Memory usage: ~1.5-2 GB
- CPU usage: 80-100% (single core)

### GPU Performance (Docker)
- Image inference: ~0.5-1 second
- Memory usage: ~2-3 GB RAM + 2 GB VRAM
- GPU usage: 40-60%

### Kubernetes (3 replicas, CPU)
- Throughput: ~60 images/minute
- Average latency: 3.2 seconds
- Memory per pod: ~2 GB

---

## 🔗 Useful Commands

### Docker

```bash
# Remove all RipCatch containers
docker rm -f $(docker ps -a | grep ripcatch | awk '{print $1}')

# Remove all RipCatch images
docker rmi $(docker images | grep ripcatch | awk '{print $3}')

# Clean up system
docker system prune -a --volumes

# View resource usage
docker stats ripcatch
```

### Kubernetes

```bash
# Delete all RipCatch resources
kubectl delete -f k8s/

# Force delete pod
kubectl delete pod <pod-name> --force --grace-period=0

# Get all resources
kubectl get all -l app=ripcatch

# Export deployment
kubectl get deployment ripcatch-deployment -o yaml > backup.yaml
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Gradio Docker Deployment](https://gradio.app/sharing-your-app/#deploying-with-docker)

---

## 🆘 Support

If you encounter issues:

1. Check logs: `docker logs ripcatch` or `kubectl logs <pod-name>`
2. Review this guide's troubleshooting section
3. Open issue: https://github.com/naga-narala/RipCatch/issues
4. Contact: sravankumar.nnv@gmail.com

---

**Last Updated**: January 27, 2026  
**Version**: 2.0  
**Author**: Sravan Kumar
