from setuptools import dist

from pathlib import Path
from setuptools import find_packages
from setuptools import setup

path = Path(__file__).parent
with open(path / 'requirements.txt') as f:
    lines = f.readlines()

install_requires = [
    line.replace('\n', '')
    for line in lines
]

setup(
    name='deep-shadows',
    description='a novel computational framework that enables the quantification of sunlight access at a world scale',
    author_email='fabiom@gmail.com',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=install_requires,
    setup_requires=[
        'setuptools>=18.0'
    ],
)
