# from setuptools import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules=cythonize("CooperativeAStarFast.pyx")
# )

from distutils.core import setup
from Cython.Build import cythonize
import numpy

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("CooperativeAStarFast.pyx"),
    include_dirs=[numpy.get_include()]
)