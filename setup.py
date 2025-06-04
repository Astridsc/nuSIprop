from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension("nuSIprop", 
                       ["nuSIprop.pyx"],
                       include_dirs=[np.get_include()],
                       libraries=["gsl", "gslcblas"],
                       extra_compile_args=["-O3", "-std=gnu++11"],
                       language="c++")]

setup(
    name="nuSIprop",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
