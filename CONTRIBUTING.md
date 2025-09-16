# Contributing to LLM Differential Privacy

Thank you for your interest in contributing to the LLM Differential Privacy project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Git
- A GitHub account
- Basic understanding of differential privacy and machine learning

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/LLMEncrption2.git
   cd LLMEncrption2
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv dp_env
   source dp_env/bin/activate  # On Windows: dp_env\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements.txt[dev]  # For development tools
   ```

5. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## 📋 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Testing

- Write tests for new features
- Ensure all existing tests pass
- Aim for good test coverage
- Test both success and failure cases

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update type hints and comments
- Include examples in docstrings

## 🔧 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write your code following the style guidelines
- Add tests for your changes
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run linting
flake8 .
black --check .
mypy .

# Run tests
pytest tests/

# Test the main functionality
python LLM_Diffrential_Privacy.py --encrypt-data --records 10
python LLM_Diffrential_Privacy.py --train --records 5 --epochs 1
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of your changes

- Detailed description of what was changed
- Why the change was made
- Any breaking changes or considerations"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the bug
2. **Steps to reproduce** the issue
3. **Expected behavior** vs actual behavior
4. **Environment details** (OS, Python version, etc.)
5. **Error messages** and logs
6. **Minimal code example** if applicable

## ✨ Feature Requests

When requesting features, please include:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** (if you have ideas)
4. **Alternative solutions** considered
5. **Additional context** or references

## 🔒 Security Considerations

This project handles sensitive data and privacy. When contributing:

- **Never commit sensitive data** (PHI, PII, API keys, etc.)
- **Follow privacy best practices** in your code
- **Test privacy guarantees** when modifying DP mechanisms
- **Document security implications** of changes
- **Use secure coding practices**

## 📊 Code Review Process

### For Contributors

1. **Self-review** your code before submitting
2. **Address feedback** promptly and constructively
3. **Be responsive** to reviewer comments
4. **Keep PRs focused** and reasonably sized
5. **Update documentation** as needed

### For Reviewers

1. **Be constructive** and respectful
2. **Focus on code quality** and correctness
3. **Check privacy implications** of changes
4. **Test the changes** if possible
5. **Provide clear feedback** and suggestions

## 🏷️ Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Security review completed
- [ ] Privacy implications reviewed

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** and helpful
- **Be professional** in all interactions

### Getting Help

- **GitHub Issues** for bug reports and feature requests
- **GitHub Discussions** for questions and general discussion
- **Pull Request comments** for code-specific questions

## 📚 Resources

### Differential Privacy
- [Differential Privacy Book](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [OpenDP](https://opendp.io/)
- [Microsoft DP Guide](https://docs.microsoft.com/en-us/azure/machine-learning/concept-differential-privacy)

### Machine Learning
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

### Development Tools
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)

## 📝 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the LLM Differential Privacy project! 🎉
