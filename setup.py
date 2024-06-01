from setuptools import find_packages, setup

setup(
    name="vsnst",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow[and-cuda]",
        "nvidia-cudnn-cu11==8.6.0.163",
    ],
)
