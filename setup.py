from setuptools import setup, find_packages

setup(
    name="kgiant_distance_predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ],
    author="ROSE",
    description="K巨星距离估计"
)