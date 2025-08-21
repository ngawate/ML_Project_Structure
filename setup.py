from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This will give list of requirments
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="ML_Project_Structure",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    author="Nikhil Gawate",
    author_email="ngawate@umich.edu",
    description="A machine learning project structure.",
    url="https://github.com/ngawate/ML_Project_Structure",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)