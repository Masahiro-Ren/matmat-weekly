#pragma once

#include "misc.h"
#include <functional>

using MatmulFn = std::function<void(const double*, const double*, double*, 
                                    const size_t, const size_t, const size_t,
                                    const size_t, const size_t, const size_t)>;

struct BenchResult{
    double t_best;
    double t_avg;
    double gflops;
    double chk;
};

BenchResult benchmark(const MatmulFn& fn,
                      const double* A, const double* B, double* C,
                      size_t BM, size_t BN, size_t BK,
                      size_t M, size_t N, size_t K,
                      int warmup, int rounds);


#ifdef USE_BLAS
BenchResult blas_benchmark(const double* A, const double* B, double* C,
                           size_t M, size_t N, size_t K,
                           int warmup, int rounds);
#endif