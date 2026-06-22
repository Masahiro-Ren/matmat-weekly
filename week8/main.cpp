#include "misc.h"
#include "matmat.h"
#include "bench.h"
#include "autotune.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <limits>

int main(int argc, char* argv[])
{
using std::vector;
using std::pair;
using std::tuple;
using std::printf;


    vector<tuple<size_t, size_t, size_t>> problem_sizes{
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096}
    };

    vector<pair<std::string, MatmulFn>> impls{
        {"ikj_ij_8x16", matmat_blocked_ikj},
    };

    const int warmup = 10;
    const int rounds = 20;
    const double tol = 1e-9;

    // ------------------ AUTO-TUNING Stage --------------------------
    const size_t Mt = 2048, Nt = 2048, Kt = 2048;   // tune at a clean-to-measure size
    vector<double> A(Mt*Kt), B(Kt*Nt), C(Mt*Nt, 0.0);
    fill_random(A.data(), Mt*Kt, -1.0, 1.0);
    fill_random(B.data(), Kt*Nt, -1.0, 1.0);

    BlockConfig best = autotune(A.data(), B.data(), C.data(), Mt, Nt, Kt, warmup, rounds);
    printf("Auto-tuned best block: %zux%zux%zu  (%.1f GFLOP/s)\n",
       best.BM, best.BN, best.BK, best.score);

    const size_t BM = best.BM;
    const size_t BN = best.BN;
    const size_t BK = best.BK;

    // Block size is fixed this week, so the label is constant.
    const std::string block = std::to_string(BM) + "x"
                            + std::to_string(BN) + "x"
                            + std::to_string(BK);
    
    // ------------------ Evaluation Stage --------------------------
    for(const auto& [M, N, K] : problem_sizes)
    {
        // The 8x16 micro-kernel tiles M by MR(8), N by NR(16) with no remainder
        // handling, so M and N must divide evenly. Skip otherwise.
        if(M % MR512_8x16 || N % NR512_8x16)
        {
            printf("\n [skip] M=%zu, N=%zu not divisible by tile (need M%%8==0, N%%16==0)\n",
                   M, N);
            continue;
        }

        vector<double> A(M * K);
        vector<double> B(K * N);
        vector<double> C(M * N, 0.0);
        vector<double> Ref(M * N);

        fill_random(A.data(), M * K, -1.0, 1.0);
        fill_random(B.data(), K * N, -1.0, 1.0);

        printf("\n ========= M=%zu, N=%zu, K=%zu, (%.1f MFLOP) =========\n",
               M, N, K, 2.0 * M * N * K / 1e6);
        //      Impl          block        best(ms)    avg(ms)   GFLOP/s   %ofBLAS   status     maxdiff
        printf("  %-12s  %-11s  %10s  %10s  %9s  %8s  %7s  %11s\n",
               "Impl", "block", "best(ms)", "avg(ms)", "GFLOP/s", "%ofBLAS", "status", "maxdiff");

#ifdef USE_BLAS
        auto blas_res = blas_benchmark(A.data(), B.data(), Ref.data(), M, N, K, warmup, rounds);
        printf("  %-12s  %-11s  %10.3e  %10.3e  %9.2f  %7.1f%%  %7s  %11s\n",
               "BLAS", "N/A", blas_res.t_best, blas_res.t_avg, blas_res.gflops, 100.0, "N/A", "N/A");
#else
        matmat_ikj(A.data(), B.data(), Ref.data(), M, N, K);
#endif

        for(auto& [name, fn] : impls)
        {
            auto res = benchmark(fn, A.data(), B.data(), C.data(), BM, BN, BK, M, N, K, warmup, rounds);

            const double diff = max_abs_diff(C.data(), Ref.data(), M * N);
            const bool ok = diff <= tol * std::max(1.0, std::abs(res.chk));
#ifdef USE_BLAS
            const double pct = blas_res.gflops > 0 ? res.gflops / blas_res.gflops * 100.0 : 0.0;
            printf("  %-12s  %-11s  %10.3e  %10.3e  %9.2f  %7.1f%%  %7s  %11.2e\n",
                    name.c_str(), block.c_str(), res.t_best, res.t_avg, res.gflops, pct, ok ? "OK" : "WRONG", diff);
#else
            printf("  %-12s  %-11s  %10.3e  %10.3e  %9.2f  %8s  %7s  %11.2e\n",
                    name.c_str(), block.c_str(), res.t_best, res.t_avg, res.gflops, "N/A", ok ? "OK" : "WRONG", diff);
#endif
        }

    }

    return 0;
}