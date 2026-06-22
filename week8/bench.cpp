#include "bench.h"

#ifdef USE_BLAS
#include <cblas.h>
#endif

BenchResult benchmark(const MatmulFn& fn,
                      const double* A, const double* B, double* C,
                      size_t BM, size_t BN, size_t BK,
                      size_t M, size_t N, size_t K,
                      int warmup, int rounds)
{
    Timer timer;
    double t_best = std::numeric_limits<double>::infinity();
    double t_total = 0.0;

    // Warmup rounds
    for(int w = 0; w < warmup; w++)
        fn(A, B, C, BM, BN, BK, M, N, K);
    
    // Measure rounds
    for(int r = 0; r < rounds; r++)
    {
        // Clean C
        std::fill_n(C, M * N, 0.0);

        timer.start();
        fn(A, B, C, BM, BN, BK, M, N, K);
        const double t = timer.get();

        if(t < t_best) t_best = t;
        t_total += t; 
    }

    BenchResult res;
    const double flop = 2.0 * double(M) * double(N) * double(K);
    res.t_best = t_best;
    res.t_avg = t_total / rounds;
    res.gflops = flop / res.t_avg / 1e9;
    res.chk = checksum(C, M * N);

    return res;
}

#ifdef USE_BLAS
BenchResult blas_benchmark(const double* A, const double* B, double* C,
                           size_t M, size_t N, size_t K,
                           int warmup, int rounds)
{
    Timer timer;
    double t_best = std::numeric_limits<double>::infinity();
    double t_total = 0.0;

    for(int w = 0; w < warmup; w++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        int(M), int(N), int(K), 1.0,
                        A, int(K), B, int(N),
                        0.0, C, int(N));
    
    for(int r = 0; r < rounds; r++)
    {
        timer.start();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        int(M), int(N), int(K), 1.0,
                        A, int(K), B, int(N),
                        0.0, C, int(N));
        const double t = timer.get();

        if(t < t_best) t_best = t;
        t_total += t; 
    }

    BenchResult res;
    const double flop = 2.0 * double(M) * double(N) * double(K);
    res.t_best = t_best;
    res.t_avg = t_total / rounds;
    res.gflops = flop / res.t_avg / 1e9;
    res.chk = checksum(C, M * N);

    return res;
}
#endif