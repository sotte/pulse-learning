# pulse-learning

## Performance

The crucial parts of the code (precomputing the feature matrices and objective
evaluations) are parallelized, which gives a significant performance boost on
multi-core machines. Also, the RELEASE target is much faster (about a factor of
10) than the DEBUG target, which is presumably due to disabling debug
checks/output on the preprocessor level and optimizing the code aggressively
(-O3).

## Dependencies

pulse-learning uses an [L-BFGS
library](http://www.chokkan.org/software/liblbfgs/) for optimizing the feature
weights. For Linux (at least Arch and Ubuntu) there are ready-made packages
available.

Ubuntu:

    sudo apt-get install libarmadillo-dev liblbfgs-dev


## Getting Started

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=DEBUG ..
    make
