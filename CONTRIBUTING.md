# Contributing to GEMEX

Thank you for your interest in contributing to GEMEX. This document describes
how to report issues, suggest improvements, and contribute code.

---

## Reporting Issues

Please use the GitHub issue tracker at:
https://github.com/utkukose/gemex/issues

When reporting a bug, include:

- GEMEX version (`import gemex; print(gemex.__version__)`)
- Python version and operating system
- Minimal reproducible example (the shortest code that shows the problem)
- Full error traceback

---

## Feature Requests

Open a GitHub issue with the label `enhancement`. Describe:

- What you want GEMEX to do
- Why it would be useful
- Whether you would like to implement it yourself

---

## Pull Requests

Before submitting a pull request:

1. Open an issue first to discuss the change
2. Fork the repository and create a feature branch
3. Follow the code style of the existing codebase (PEP 8, docstrings on all
   public functions)
4. Add or update tests in `tests/test_gemex.py`
5. Run the test suite: `python -m pytest tests/`
6. Ensure all existing tests still pass

---

## Development Setup

```bash
git clone https://github.com/utkukose/gemex
cd gemex
pip install -e ".[dev]"
python -m pytest tests/
```

---

## Areas Where Contributions Are Most Welcome

- SLIC superpixel segmentation as alternative to grid patches
- Additional dataset loaders in example scripts
- Performance optimisation (vectorised FIM computation)
- Additional language support in `language_detail` narrative generation

---

## Code of Conduct

Be respectful and constructive in all interactions.

---

## Contact

Dr. Utku Kose — utkukose@gmail.com — https://www.utkukose.com
ORCID: 0000-0002-9652-6415
