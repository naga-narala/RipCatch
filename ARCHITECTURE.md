# 🏗️ RipCatch v2.0 - Container Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RipCatch v2.0 Deployment                         │
│                        Complete Containerization                         │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │  Developer   │
                              │   Commits    │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────────┐
                              │  GitHub Actions  │
                              │   CI/CD Build    │
                              └──────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                 │
            ┌───────▼────────┐              ┌────────▼─────────┐
            │  CPU Image     │              │   GPU Image      │
            │  1.2 GB        │              │   4.5 GB         │
            │  Python 3.10   │              │   CUDA 11.8      │
            └───────┬────────┘              └────────┬─────────┘
                    │                                │
                    └────────────────┬───────────────┘
                                     │
                         ┌───────────▼────────────┐
                         │    Docker Hub          │
                         │ naga-narala/ripcatch   │
                         └───────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
         ┌──────────▼──────┐  ┌──────▼──────┐  ┌────▼────────┐
         │  Local Docker   │  │   Docker    │  │ Kubernetes  │
         │   Standalone    │  │  Compose    │  │   Cluster   │
         └────────┬────────┘  └──────┬──────┘  └────┬────────┘
                  │                  │               │
         ┌────────▼────────┐  ┌──────▼──────┐  ┌────▼────────┐
         │  Port 7860      │  │  CPU: 7860  │  │ LoadBalancer│
         │  Single         │  │  GPU: 7861  │  │   Port 80   │
         │  Container      │  │  Services   │  │   SSL/TLS   │
         └─────────────────┘  └─────────────┘  └─────────────┘


═══════════════════════════════════════════════════════════════════════════
                           DEPLOYMENT METHODS
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. DOCKER STANDALONE                                                     │
│    docker run -d -p 7860:7860 naga-narala/ripcatch:latest-cpu          │
│    ✓ Fastest to deploy      ✓ Minimal config     ✓ Good for testing   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. DOCKER COMPOSE                                                        │
│    docker-compose up -d ripcatch-cpu                                    │
│    ✓ Multi-service          ✓ Volume management  ✓ Easy restart        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. KUBERNETES                                                            │
│    kubectl apply -f k8s/deployment.yaml                                 │
│    ✓ Production-ready       ✓ Auto-scaling       ✓ High availability   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. HUGGING FACE SPACES                                                   │
│    https://huggingface.co/spaces/sravankumarnnv/ripcatch               │
│    ✓ Already deployed       ✓ Free hosting       ✓ GPU available       │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
                          IMAGE ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                        CPU IMAGE (Multi-stage)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Stage 1: Base                                                           │
│    ├─ python:3.10-slim                                                   │
│    ├─ System dependencies (OpenGL, libgomp)                              │
│    └─ Working directory: /app                                            │
│                                                                          │
│  Stage 2: Dependencies                                                   │
│    ├─ requirements.txt                                                   │
│    ├─ PyTorch CPU                                                        │
│    ├─ Ultralytics YOLOv8                                                 │
│    └─ Gradio 4.0.0                                                       │
│                                                                          │
│  Stage 3: Application                                                    │
│    ├─ app.py (Gradio interface)                                          │
│    ├─ best.pt (52 MB model)                                              │
│    ├─ Non-root user (security)                                           │
│    └─ Health checks                                                      │
│                                                                          │
│  Exposed Port: 7860                                                      │
│  Final Size: ~1.2 GB                                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        GPU IMAGE (CUDA-enabled)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Base: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04                    │
│    ├─ CUDA 11.8                                                          │
│    ├─ cuDNN 8                                                            │
│    ├─ Python 3.10                                                        │
│    ├─ PyTorch GPU (CUDA 11.8)                                            │
│    ├─ Application files                                                  │
│    └─ GPU optimization flags                                             │
│                                                                          │
│  Environment:                                                            │
│    ├─ CUDA_HOME=/usr/local/cuda                                          │
│    ├─ NVIDIA_VISIBLE_DEVICES=all                                         │
│    └─ NVIDIA_DRIVER_CAPABILITIES=compute,utility                         │
│                                                                          │
│  Exposed Port: 7860                                                      │
│  Final Size: ~4.5 GB                                                     │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
                       KUBERNETES ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                         Internet / Users                                 │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
         ┌────────────▼─────────────┐
         │  NGINX Ingress Controller │
         │  SSL/TLS Termination      │
         │  ripcatch.yourdomain.com  │
         └────────────┬──────────────┘
                      │
         ┌────────────▼─────────────┐
         │  Kubernetes Service       │
         │  Type: LoadBalancer       │
         │  Port: 80 → 7860         │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────────────────┐
         │         Deployment                     │
         │  ┌────────┐  ┌────────┐  ┌────────┐  │
         │  │ Pod 1  │  │ Pod 2  │  │ Pod N  │  │
         │  │        │  │        │  │        │  │
         │  │ CPU:1  │  │ CPU:1  │  │ CPU:1  │  │
         │  │ MEM:2G │  │ MEM:2G │  │ MEM:2G │  │
         │  │        │  │        │  │        │  │
         │  │ App    │  │ App    │  │ App    │  │
         │  │ Model  │  │ Model  │  │ Model  │  │
         │  └────────┘  └────────┘  └────────┘  │
         │                                        │
         │  Auto-scaling: 2-10 replicas          │
         │  CPU threshold: 70%                   │
         └────────────┬──────────────────────────┘
                      │
         ┌────────────▼─────────────┐
         │  PersistentVolumeClaim   │
         │  Model Storage (1Gi)     │
         │  ReadOnlyMany            │
         └──────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
                           CI/CD WORKFLOW
═══════════════════════════════════════════════════════════════════════════

Developer Workflow:
──────────────────

1. Code Changes
   └─> git commit
       └─> git push origin main

2. GitHub Actions Triggered
   ├─> Checkout code
   ├─> Setup Docker Buildx
   ├─> Login to Docker Hub
   │
   ├─> Build CPU Image ─────────┐
   │   └─> Tag: latest-cpu       │
   │                              ├─> Push to Docker Hub
   ├─> Build GPU Image ─────────┤
   │   └─> Tag: latest-gpu       │
   │                              │
   └─> Security Scan ────────────┘
       └─> Trivy vulnerability scan

3. Images Available
   ├─> docker pull naga-narala/ripcatch:latest-cpu
   └─> docker pull naga-narala/ripcatch:latest-gpu

4. Deploy
   ├─> Local: docker run / docker-compose
   ├─> Cloud: kubectl apply
   └─> HF: Already deployed


═══════════════════════════════════════════════════════════════════════════
                        SECURITY LAYERS
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ Container Security                                                       │
│  ✓ Non-root user (UID 1000)                                             │
│  ✓ Read-only filesystem where possible                                  │
│  ✓ No sensitive data in images                                          │
│  ✓ Health checks for reliability                                        │
│  ✓ Resource limits (CPU, memory)                                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Network Security                                                         │
│  ✓ Ingress with SSL/TLS (cert-manager)                                  │
│  ✓ Service-to-service encryption                                        │
│  ✓ Network policies (isolation)                                         │
│  ✓ Rate limiting on ingress                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ CI/CD Security                                                           │
│  ✓ Trivy vulnerability scanning                                         │
│  ✓ GitHub Secrets for credentials                                       │
│  ✓ Signed commits (optional)                                            │
│  ✓ SARIF reports to Security tab                                        │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
                       MONITORING & OBSERVABILITY
═══════════════════════════════════════════════════════════════════════════

Container Logs:
  docker logs -f ripcatch
  kubectl logs -f deployment/ripcatch-deployment

Health Checks:
  HTTP GET http://localhost:7860/
  Interval: 30s | Timeout: 10s | Retries: 3

Resource Monitoring:
  docker stats ripcatch
  kubectl top pods

Metrics (Kubernetes):
  Prometheus integration ready
  Grafana dashboards available
  Alert manager configured
