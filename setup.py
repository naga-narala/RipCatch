#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RipCatch - AI-Powered Rip Current Detection System
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "AI-powered rip current detection system using YOLOv8"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ]

# Development requirements
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "flake8>=6.1.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
]

setup(
    # Package metadata
    name="ripcatch",
    version="2.0.0",
    author="Sravan Kumar (naga-narala)",
    author_email="sravankumar.nnv@gmail.com",
    description="AI-powered rip current detection system using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naga-narala/RipCatch",
    project_urls={
        "Bug Reports": "https://github.com/naga-narala/RipCatch/issues",
        "Source": "https://github.com/naga-narala/RipCatch",
        "Documentation": "https://github.com/naga-narala/RipCatch/wiki",
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    
    # Package data
    package_data={
        "ripcatch": ["*.yaml", "*.yml", "*.json"],
    },
    include_package_data=True,
    
    # Entry points (console scripts)
    entry_points={
        "console_scripts": [
            "ripcatch-detect=ripcatch.cli:detect_cli",
            "ripcatch-train=ripcatch.cli:train_cli",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords=[
        "rip-current",
        "detection",
        "yolov8",
        "computer-vision",
        "deep-learning",
        "beach-safety",
        "ocean",
        "pytorch",
    ],
    
    # License
    license="MIT",
    
    # Zip safe
    zip_safe=False,
)

"""
Installation Instructions:
==========================

1. Install in development mode (recommended for development):
   pip install -e .

2. Install with all dependencies:
   pip install -e .[all]

3. Install with development dependencies:
   pip install -e .[dev]

4. Install from source:
   pip install .

5. Install from GitHub:
   pip install git+https://github.com/naga-narala/RipCatch.git

Uninstallation:
===============

pip uninstall ripcatch

Usage After Installation:
=========================

# As a package
from ripcatch import detect_rip_current

# Using command-line tools
ripcatch-detect --image path/to/image.jpg
ripcatch-train --data path/to/data.yaml

Notes:
======

- This package requires PyTorch to be installed separately
- For GPU support, install PyTorch with CUDA:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

- Model weights are not included in the package
  Download separately from: https://github.com/naga-narala/RipCatch/releases
"""

