#include "misc.h"
#include "matmat.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <functional>
#include <string>
#include <limits>

#ifdef USE_BLAS
#include <cblas.h>
#endif

using MatmulFn = std::function<void(const double*, const double*, double*, 
                                    const size_t, const size_t, const size_t)>;

struct BenchResult{
    double t_best;
    double t_avg;
    double gflops;
    double chk;
};

BenchResult benchmark(const MatmulFn& fn,
                      const double* A, const double* B, double* C,
                      size_t M, size_t N, size_t K,
                      int warmup, int rounds)
{
    Timer timer;
    double t_best = std::numeric_limits<double>::infinity();
    double t_total = 0.0;

    // Warmup rounds
    for(int w = 0; w < warmup; w++)
        fn(A, B, C, M, N, K);
    
    // Measure rounds
    for(int r = 0; r < rounds; r++)
    {
        // Clean C
        std::fill_n(C, M * N, 0.0);

        timer.start();
        fn(A, B, C, M, N, K);
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

int main(int argc, char* argv[])
{
using std::vector;
using std::pair;
using std::tuple;
using std::printf;

    vector<tuple<size_t, size_t, size_t>> problem_sizes{
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };

    vector<pair<std::string, MatmulFn>> impls{
        {"ij_micro4x4", matmat_micro4x4},
        {"ij_micro4x3", matmat_micro4x3}
    };

    const int warmup = 10;
    const int rounds = 20;
    const double tol = 1e-9;

    for(const auto& [M, N, K] : problem_sizes)
    {

        vector<double> A(M * K);
        vector<double> B(K * N);
        vector<double> C(M * N, 0.0);
        vector<double> Ref(M * N);

        fill_random(A.data(), M * K, -1.0, 1.0);
        fill_random(B.data(), K * N, -1.0, 1.0);

        printf("\n ========= M=%zu, N=%zu, K=%zu, (%.1f MFLOP) =========\n",
               M, N, K, 2.0 * M * N * K / 1e6);
        //      Impl          best(ms)    avg(ms)   GFLOP/s   %ofBLAS   status     maxdiff
        printf("  %-12s  %10s  %10s  %9s  %8s  %7s  %11s\n",
               "Impl", "best(ms)", "avg(ms)", "GFLOP/s", "%ofBLAS", "status", "maxdiff");

#ifdef USE_BLAS
        auto blas_res = blas_benchmark(A.data(), B.data(), Ref.data(), M, N, K, warmup, rounds);
        printf("  %-12s  %10.4f  %10.4f  %9.2f  %7.1f%%  %7s  %11s\n",
               "BLAS", blas_res.t_best, blas_res.t_avg, blas_res.gflops, 100.0, "N/A", "N/A");
#else
        matmat_ikj(A.data(), B.data(), Ref.data(), M, N, K);
#endif

        for(auto& [name, fn] : impls)
        {
            auto res = benchmark(fn, A.data(), B.data(), C.data(), M, N, K, warmup, rounds);

            const double diff = max_abs_diff(C.data(), Ref.data(), M * N);
            const bool ok = diff <= tol * std::max(1.0, std::abs(res.chk));
#ifdef USE_BLAS
            const double pct = blas_res.gflops > 0 ? res.gflops / blas_res.gflops * 100.0 : 0.0;
            printf("  %-12s  %10.4f  %10.4f  %9.2f  %7.1f%%  %7s  %11.2e\n",
                   name.c_str(), res.t_best, res.t_avg, res.gflops, pct, ok ? "OK" : "WRONG", diff);
#else
            printf("  %-12s  %10.4f  %10.4f  %9.2f  %8s  %7s  %11.2e\n",
                   name.c_str(), res.t_best, res.t_avg, res.gflops, "N/A", ok ? "OK" : "WRONG", diff);
#endif
        }

    }

    return 0;
}