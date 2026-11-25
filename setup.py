#!/usr/bin/env python3
"""Setup script for LLM Differential Privacy project."""

from setuptools import find_packages, setup


def read_readme() -> str:
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def read_requirements() -> list[str]:
    reqs: list[str] = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            reqs.append(line)
    return reqs


def read_version() -> str:
    with open("LLM_Diffrential_Privacy.py", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "1.0.0"


setup(
    name="llm-differential-privacy",
    version=read_version(),
    author="LLM Differential Privacy Team",
    description="A framework for training LLMs with differential privacy using VaultGemma",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/oyesanyf/VaultGemmaSampleCode",
    packages=find_packages(exclude=("tests", "tests.*")),
    py_modules=["LLM_Diffrential_Privacy", "test_model_queries"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=26.3.1",
            "flake8>=7.0.0",
            "mypy>=1.11.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-dp-train=LLM_Diffrential_Privacy:main",
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
    ],
    license="MIT",
    platforms=["any"],
)
