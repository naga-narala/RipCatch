# Contributing to RipCatch

Thank you for your interest in contributing to RipCatch! This document provides guidelines and instructions for contributing to the project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone. We expect all contributors to:

- ✅ Be respectful and considerate
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what is best for the community
- ✅ Show empathy towards others

### Unacceptable Behavior

- ❌ Harassment, trolling, or discriminatory comments
- ❌ Publishing others' private information
- ❌ Spam or off-topic discussions
- ❌ Any conduct that creates an intimidating environment

---

## 🤝 How Can I Contribute?

### 1. 🐛 Reporting Bugs

Found a bug? Help us fix it!

**Before submitting:**
- Search existing [issues](https://github.com/naga-narala/RipCatch/issues) to avoid duplicates
- Test with the latest version
- Gather as much information as possible

**Bug Report Template:**

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or error logs.

**Environment:**
 - OS: [e.g., Windows 11, Ubuntu 22.04]
 - Python Version: [e.g., 3.10.12]
 - PyTorch Version: [e.g., 2.0.1]
 - CUDA Version: [e.g., 11.8]
 - GPU: [e.g., RTX 3080]

**Additional context**
Any other information about the problem.
```

---

### 2. 💡 Suggesting Features

Have an idea to improve RipCatch?

**Feature Request Template:**

```markdown
**Feature Description**
Clear description of the proposed feature.

**Problem it Solves**
Explain what problem this feature addresses.

**Proposed Solution**
Describe how you envision this working.

**Alternatives Considered**
Other approaches you've thought about.

**Additional Context**
Any relevant examples, mockups, or references.
```

---

### 3. 📊 Contributing Data

Help improve model accuracy!

**Dataset Contributions:**
- Original beach/rip current images or videos
- Properly labeled data in YOLO format
- Diverse conditions (weather, time of day, locations)

**Data Requirements:**
- High resolution (1920×1080 or higher)
- Clear visibility of water conditions
- Diverse geographic locations
- Proper attribution and usage rights

**How to Submit:**
1. Open an issue with `[Dataset]` tag
2. Provide dataset description and samples
3. Include licensing information
4. Wait for review and approval

---

### 4. 🔧 Code Contributions

Ready to write code? Follow these guidelines:

#### Areas for Contribution

**High Priority:**
- 🔥 Performance optimizations
- 🔥 Mobile deployment support
- 🔥 Real-time processing improvements
- 🔥 Documentation enhancements

**Medium Priority:**
- 🌟 New augmentation techniques
- 🌟 Additional export formats
- 🌟 Web interface development
- 🌟 Testing and validation scripts

**Low Priority (but welcome):**
- ⭐ Code refactoring
- ⭐ Style improvements
- ⭐ Minor bug fixes

---

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork repository on GitHub, then:
git clone https://github.com/naga-narala/RipCatch.git
cd RipCatch

# Add upstream remote
git remote add upstream https://github.com/naga-narala/RipCatch.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 isort mypy pre-commit
```

### 3. Configure Git Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Hooks will run automatically on commit
# To run manually:
pre-commit run --all-files
```

### 4. Verify Setup

```bash
# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test Ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics OK')"

# Run tests
pytest tests/
```

---

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# ✅ Good
def detect_rip_current(image_path: str, confidence: float = 0.25) -> list:
    """
    Detect rip currents in an image.
    
    Args:
        image_path: Path to input image
        confidence: Detection confidence threshold (0-1)
        
    Returns:
        List of detection dictionaries
    """
    model = YOLO('best.pt')
    results = model(image_path, conf=confidence)
    return results[0].boxes.data.tolist()

# ❌ Bad
def detect(img,conf=0.25):  # Missing type hints, no docstring
    m=YOLO('best.pt')  # Poor variable names
    return m(img,conf=conf)[0].boxes.data.tolist()
```

### Key Conventions

**Naming:**
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names (avoid single letters except in loops)

**Documentation:**
- Docstrings for all public functions/classes
- Type hints for function arguments and returns
- Inline comments for complex logic

**Code Organization:**
- Group related functions
- Keep functions focused (single responsibility)
- Avoid deep nesting (max 3-4 levels)

**Imports:**
```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import torch
from ultralytics import YOLO

# Local
from ripcatch.utils import load_config
```

---

## 🧪 Testing Guidelines

### Writing Tests

**Test Structure:**
```python
# tests/test_detection.py
import pytest
from ultralytics import YOLO

@pytest.fixture
def model():
    """Load model for testing."""
    return YOLO('RipCatch-v2.0/Model/weights/best.pt')

def test_image_detection(model):
    """Test single image detection."""
    results = model('Testing/Mixed/RIP1.webp')
    assert len(results) == 1
    assert results[0].boxes is not None

def test_confidence_threshold(model):
    """Test confidence filtering."""
    results_low = model('test.jpg', conf=0.1)
    results_high = model('test.jpg', conf=0.8)
    assert len(results_low[0].boxes) >= len(results_high[0].boxes)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ripcatch tests/

# Run specific test file
pytest tests/test_detection.py

# Run specific test
pytest tests/test_detection.py::test_image_detection

# Verbose output
pytest -v
```

### Test Requirements

- ✅ All new features must have tests
- ✅ Maintain >80% code coverage
- ✅ Tests must pass on CI/CD before merge
- ✅ Include both positive and negative test cases

---

## 💬 Commit Message Guidelines

We follow the **Conventional Commits** specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples

```bash
# Feature
git commit -m "feat(detection): add real-time video stream support"

# Bug fix
git commit -m "fix(training): resolve CUDA out-of-memory error with batch size 32"

# Documentation
git commit -m "docs(readme): update installation instructions for Windows"

# Performance
git commit -m "perf(inference): optimize image preprocessing for 2x speedup"

# Multi-line
git commit -m "feat(export): add TensorRT export support

- Implement TensorRT conversion
- Add INT8 quantization option
- Update export documentation

Closes #123"
```

### Rules

- ✅ Use imperative mood ("add" not "added")
- ✅ Don't capitalize first letter
- ✅ No period at the end of subject
- ✅ Keep subject line under 50 characters
- ✅ Wrap body at 72 characters
- ✅ Reference issues/PRs in footer

---

## 🔄 Pull Request Process

### Before Submitting

1. **Sync with upstream:**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run quality checks:**
```bash
# Format code
black .
isort .

# Check style
flake8 .

# Type checking
mypy ripcatch/

# Run tests
pytest
```

3. **Update documentation:**
   - Update README if adding features
   - Add docstrings to new functions
   - Update CHANGELOG.md

### Creating Pull Request

1. **Push to your fork:**
```bash
git push origin feature/your-feature
```

2. **Open PR on GitHub:**
   - Use clear, descriptive title
   - Fill out PR template completely
   - Reference related issues
   - Add screenshots/demos if applicable

3. **PR Template:**

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
Describe tests you ran and their results.

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to demonstrate changes.
```

### Review Process

1. **Automated checks** run on all PRs
2. **Code review** by maintainers (typically 1-3 days)
3. **Address feedback** and push updates
4. **Approval and merge** by project maintainer

### After Merge

- ✅ Delete your feature branch
- ✅ Sync your fork with upstream
- ✅ Check the updated documentation

---

## 🐛 Issue Reporting

### Issue Labels

We use labels to categorize issues:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested
- `wontfix` - This will not be worked on
- `duplicate` - Duplicate of existing issue
- `invalid` - Not relevant or incorrect

### Issue Priority

- `P0: Critical` - Blocks core functionality
- `P1: High` - Important but not blocking
- `P2: Medium` - Nice to have
- `P3: Low` - Minor improvements

---

## 🎯 Development Roadmap

### Current Sprint (v2.1)
- [ ] Real-time video stream optimization
- [ ] Multi-camera tracking support
- [ ] Mobile app development
- [ ] API endpoint creation

### Backlog
- Enhanced temporal analysis
- Weather integration
- Crowd density estimation
- Multi-language support

**See [GitHub Projects](https://github.com/naga-narala/RipCatch/projects) for detailed roadmap.**

---

## 📚 Additional Resources

### Documentation
- [README.md](README.md) - Project overview
- [QUICK_START.md](QUICK_START.md) - Quick setup guide
- [TRAINING_SUMMARY_REPORT.md](RipCatch-v2.0/Documentation/TRAINING_SUMMARY_REPORT.md) - Training details

### External Links
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Python PEP 8 Style Guide](https://pep8.org/)

---

## 🙏 Recognition

All contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Mentioned in release notes
- Acknowledged in project documentation

**Top contributors may receive:**
- Direct commit access
- Special badges/recognition
- Invitation to project planning discussions

---

## 📞 Getting Help

**Need assistance?**
- 💬 [GitHub Discussions](https://github.com/naga-narala/RipCatch/discussions)
- 📧 Email: sravankumar.nnv@gmail.com
- 📖 [Documentation](https://github.com/naga-narala/RipCatch/wiki)

---

<div align="center">

**Thank you for contributing to RipCatch! 🌊**

Every contribution, no matter how small, helps make beaches safer.

</div>

