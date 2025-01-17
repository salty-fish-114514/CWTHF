#python D:\sci\haloidentify\cython\setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize(r"D:\sci\haloidentify\cython\halo_segment.pyx"),
      include_dirs=[np.get_include()],requires=['Cython','numpy'])