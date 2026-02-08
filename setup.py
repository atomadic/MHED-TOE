from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mhed_toe",
    version="1.0.0",
    author="Thomas Ralph Colvin IV",
    author_email="atomadic@proton.me",
    description="Monadic-Hex Entropic Dynamics Theory of Everything",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atomadic/MHED-TOE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "qutip>=4.7.1",
        "networkx>=3.1",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "sympy>=1.9",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=5.0",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mhed-simulate=mhed_toe.cli:main",
        ],
    },
)
