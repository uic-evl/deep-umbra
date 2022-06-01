from Cython.Build import cythonize
import warnings
from pathlib import Path
import numpy
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
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize('cutil/functions.pyx'),
)
