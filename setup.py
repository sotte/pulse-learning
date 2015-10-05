from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext = Extension(
    name="pypulse",
    sources=["pypulse.pyx"],
    language="c++",
    extra_compile_args=["-std=c++11", '-fPIC'],
    include_dirs=[r'.'],
    library_dirs=[r'.'],
    libraries=['simplepulse', 'lbfgs'],  # use build/lisimplebpulse.a
)

setup(
    name='pypulse',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext],
)
