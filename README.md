# pulse-learning

## Performance

The crucial parts of the code (precomputing the feature matrices and objective
evaluations) are parallelized, which gives a significant performance boost on
multi-core machines. Also, the RELEASE target is much faster (about a factor of
10) than the DEBUG target, which is presumably due to disabling debug
checks/output on the preprocessor level and optimizing the code aggressively
(-O3).
