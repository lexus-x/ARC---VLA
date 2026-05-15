from setuptools import setup, find_packages

setup(
    name="apex-vla",
    version="0.1.0",
    description="APEX: Adaptive Pruning and Extraction for Vision-Language-Action Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="lexus-x",
    url="https://github.com/lexus-x/ARC---VLA",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
