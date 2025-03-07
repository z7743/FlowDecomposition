from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="flow_decomposition",             
    version="0.0.1",                        
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},               
    include_package_data=True,
    python_requires=">=3.9",              
    install_requires=[ 
        "numpy",
        "pandas",
        "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)