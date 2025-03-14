# setup.py

from setuptools import setup, find_packages

setup(
    name="skolyoz_tespit",
    version="0.1.0",
    description="X-ray görüntülerinden skolyoz tespiti ve Cobb açısı hesaplama",
    author="AI Assistant",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.4.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "tqdm>=4.62.0",
        "scikit-image>=0.18.0",
        "scipy>=1.7.0",
        "pillow>=8.3.0",
    ],
    entry_points={
        'console_scripts': [
            'skolyoz-tespit=src.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)