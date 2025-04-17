from setuptools import find_packages, setup
from typing import List

hypen_e_dot = "-e ."
def get_requirements(file_path:str)->List[str]:

    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

    return requirements

setup(
    name='MLProject',
    version='0.0.1',
    author='nikhil',
    author_email='kmnikhil000@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )