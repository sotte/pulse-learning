from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext = Extension(
    name="pypulse",
    sources=["pypulse.pyx"],
    language="c++",
    extra_compile_args=['-std=c++11', '-fPIC', '-fopenmp'],
    include_dirs=[r'.'],
    library_dirs=[r'build/', r'/usr/lib/'],
    libraries=['simplepulse', 'lbfgs', 'gomp', 'armadillo'],
)

setup(
    name='pypulse',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext],
)
