# pulse-learning

## Performance

The crucial parts of the code (precomputing the feature matrices and objective
evaluations) are parallelized, which gives a significant performance boost on
multi-core machines. Also, the RELEASE target is much faster (about a factor of
10) than the DEBUG target, which is presumably due to disabling debug
checks/output on the preprocessor level and optimizing the code aggressively
(-O3).

## Dependencies

`pulse-learning` uses the
[L-BFGS library](http://www.chokkan.org/software/liblbfgs/)
for optimizing the feature weights. For Linux (at least Arch and Ubuntu) there
are ready-made packages available.

Ubuntu:

    sudo apt-get install libarmadillo-dev liblbfgs-dev cython


## Getting Started

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=DEBUG ..
    make


## The Python Interface

After creating the static lib `libsimplepulse.a` (see "Getting Started") you
can create and test the python interface:

    ./create_python_interface.sh

Or manually run the minimal example:

    python test_pypulse.py


### Internals

`SimpleInterface.cpp` is, you guessed it, a simple cpp interface around
`TemporallyExtendedModel.h` which hides custom types and c++11-goodness
(or c++11-pain in the context of cython)
to ease wrapping with cython.

`pypulse.pyx` defines the mapping from cpp (SimpleInterface.cpp) to python.

Edit `pypulse.pyx` and `SimpleInterface.cpp` to add more functionality!

The `setup.py` creates a `pypulse.cpp` from the `pypulse.pyx` file, creates
`pypulse.so`, and links against `libsimplepulse.a`.
