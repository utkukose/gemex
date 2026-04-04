# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.0
"""
setup.py — Legacy fallback for pip < 21.3.
For modern pip, pyproject.toml is the authoritative build configuration.

Copyright (c) 2026 Dr. Utku Kose
Licensed under the MIT License — see LICENSE for details.
"""
from setuptools import setup, find_packages

setup(
    name             = "gemex",
    version          = "1.2.0",
    description      = "GEMEX: Geodesic Entropic Manifold Explainability — "
                       "a novel model-agnostic XAI framework grounded in "
                       "Riemannian information geometry",
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    author           = "Dr. Utku Kose",
    author_email     = "utkukose@gmail.com",
    url              = "https://github.com/utkukose/gemex",
    python_requires  = ">=3.8",
    packages         = find_packages(),
    install_requires = [
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "pandas>=1.3",
        "tqdm>=4.60",
    ],
    extras_require = {
        "torch": ["torch>=1.10"],
        "tf":    ["tensorflow>=2.6"],
        "full":  ["torch>=1.10", "tensorflow>=2.6",
                  "plotly>=5.0", "seaborn>=0.11"],
        "dev":   ["pytest>=7.0", "black", "mypy", "flake8"],
    },
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    license = "MIT",
)
