#!/usr/bin/env python3
"""
Setup script for LLM Differential Privacy project.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version
def read_version():
    with open("LLM_Diffrential_Privacy.py", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="llm-differential-privacy",
    version=read_version(),
    author="LLM Differential Privacy Team",
    author_email="your-email@example.com",
    description="A comprehensive framework for training LLMs with differential privacy using VaultGemma",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LLMEncrption2",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/LLMEncrption2/issues",
        "Source": "https://github.com/yourusername/LLMEncrption2",
        "Documentation": "https://github.com/yourusername/LLMEncrption2/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10,<3.12",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "black>=23.0.0,<25.0.0",
            "flake8>=6.0.0,<8.0.0",
            "mypy>=1.0.0,<2.0.0",
            "pre-commit>=3.0.0,<4.0.0",
        ],
        "secure": [
            "torchcsprng>=0.2.0,<1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0,<8.0.0",
            "sphinx-rtd-theme>=1.0.0,<3.0.0",
            "myst-parser>=1.0.0,<3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-dp-train=LLM_Diffrential_Privacy:main",
            "llm-dp-query=query_model:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "differential-privacy",
        "machine-learning",
        "llm",
        "vaultgemma",
        "privacy",
        "security",
        "nlp",
        "transformers",
        "pytorch",
    ],
    license="MIT",
    platforms=["any"],
)
