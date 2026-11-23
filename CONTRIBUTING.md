# CONTRIBUTING TO NEARMISS

Thank you for your interest in contributing to **NEARMISS: Predictive Industrial Safety Intelligence System**!

We welcome contributions from data scientists, machine learning engineers, safety professionals, and backend developers.

---

## 1. Development Setup

### 1.1 Prerequisites
- Python 3.10+ (tested up to Python 3.14)
- Git

### 1.2 Installation
```bash
# Clone repository
git clone https://github.com/your-org/nearmiss.git
cd nearmiss

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 2. Workflow Guidelines

### 2.1 Branching Convention
- Feature: `feat/feature-name`
- Bugfix: `fix/bug-description`
- Documentation: `docs/doc-update`
- Optimization: `perf/optimization-target`

### 2.2 Code Quality & Formatting
Ensure all code passes formatting checks:
```bash
# Code formatting
black src/ app/ tests/
isort src/ app/ tests/

# Linting
flake8 src/ app/ tests/ --max-line-length=120
```

### 2.3 Running Tests
All pull requests must pass the test suite:
```bash
python -m pytest tests/ -v
```

---

## 3. Data Integrity & Safety Ethics Policy
1. **Never commit raw proprietary plant data or worker personally identifiable information (PII)**.
2. **Never claim synthetic benchmark data is real-world proprietary evidence**.
3. **Always preserve temporal leakage safeguards**: any new feature engineering must use past-window rolling statistics without future lookahead.
4. **Always clearly present model outputs as decision-support risk estimates**, not guaranteed deterministic outcomes.

Thank you for helping make industrial workplaces safer!
