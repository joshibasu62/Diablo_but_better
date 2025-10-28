from setuptools import setup, find_packages

setup(
    name="diablo_env",               
    version="0.1.0",                
    packages=find_packages(),        
    install_requires=[
        "gymnasium>=0.30",           
        "pybullet>=3.0",            
        "numpy>=1.25"                
    ],
    include_package_data=True,       
    description="Custom Gymnasium environment for Diablo robot simulation in PyBullet",
    author="Basanta Joshi",
    license="MIT",
    url="https://github.com/joshibasu62/diablo_env", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
