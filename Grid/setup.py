from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([Extension("grid_fast", ["grid_fast.pyx"])]),
    include_dirs=[numpy.get_include()]
)