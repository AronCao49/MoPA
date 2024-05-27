from setuptools import setup
from setuptools import find_packages

exclude_dirs = ("configs",)

# for install, do: pip install -ve .

setup(
    name='mopa',
    version="0.0.1",
    url="https://github.com/AronCao49/MoPA",
    description="MoPA: Multi-Modal Prior Aided Domain Adaptation for 3D Semantic Segmentation",
    install_requires=[
        'yacs', 
        'nuscenes-devkit', 
        'tabulate', 
        'opencv-python==4.5.5.64', 
        'Werkzeug==2.2.2', 
        'scikit-image',
        'torch-ema',
        'progress',
        'openpyxl'],
    packages=find_packages(exclude=exclude_dirs),
)