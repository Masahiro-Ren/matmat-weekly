#include "autotune.h"
#include "matmat.h"
#include "bench.h"
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <cstdio>

std::vector<size_t> valid_values(size_t dim, size_t tile)
{
    std::vector<size_t> vals;
    for(size_t b = tile; b < dim; b += tile)
        if(dim % b == 0) vals.push_back(b);

    return vals;
}

double evaluate(const double* A, const double* B, double* C,
                size_t BM, size_t BN, size_t BK,
                size_t M, size_t N, size_t K,
                int warmup, int rounds)
{
    return benchmark(matmat_blocked_ikj, A, B, C,
                     BM, BN, BK,
                     M, N, K,
                     warmup, rounds).gflops;
}


BlockConfig autotune(const double* A, const double* B, double* C,
                     size_t M, size_t N, size_t K,
                     int warmup, int rounds, SAParams param)
{
    // Candidate block sizes per dimension: [0]=BM, [1]=BN, [2]=BK
    std::array<std::vector<size_t>, 3> vals{
        valid_values(M, MR512_8x16),
        valid_values(N, NR512_8x16),
        valid_values(K, 8),
    };
    for(const auto& v : vals)
        if(v.empty()) return BlockConfig{};   // no valid block size for some dimension

    RandomValue<double> uni(param.seed);
    uni.setrange(0.0, 1.0);
    RandomValue<long> rng(param.seed + 1);

    // Draw a uniform index in [0, n).
    auto pick = [&](long n) { rng.setrange(0, n - 1); return rng.getval(); };
    // Benchmark the configuration described by an index triple.
    auto score = [&](const std::array<long, 3>& x) {
        return evaluate(A, B, C, vals[0][x[0]], vals[1][x[1]], vals[2][x[2]],
                        M, N, K, warmup, rounds);
    };

    std::array<long, 3> cur{ pick((long)vals[0].size()),
                             pick((long)vals[1].size()),
                             pick((long)vals[2].size()) };
    double cur_score = score(cur);

    std::array<long, 3> best = cur;
    double best_score = cur_score;

    std::printf("[tune] init   %4zu x%4zu x%4zu  -> %8.2f GFLOP/s\n",
                vals[0][cur[0]], vals[1][cur[1]], vals[2][cur[2]], cur_score);

    int iter = 0;
    for(double T = param.T0; T > param.Tmin; T *= param.alpha)
    {
        for(int it = 0; it < param.per_T; it++)
        {
            // Perturb one randomly chosen dimension by +-1 (clamped to range).
            std::array<long, 3> next = cur;
            long dim  = pick(3);
            long step = (uni.getval() < 0.5) ? -1 : 1;
            next[dim] = std::clamp<long>(cur[dim] + step, 0, (long)vals[dim].size() - 1);

            double cand  = score(next);
            double delta = cand - cur_score;

            // Accept improvements always, worse moves with Metropolis probability.
            bool accept = (delta >= 0.0) || (uni.getval() < std::exp(delta / T));
            if(accept)
            {
                cur = next;
                cur_score = cand;
                if(cur_score > best_score)
                {
                    best_score = cur_score;
                    best = cur;
                }
            }

            std::printf("[tune] %3d  T=%7.2f  %4zu x%4zu x%4zu  %8.2f GFLOP/s  d=%+7.2f  %-6s  best=%8.2f\n",
                        ++iter, T,
                        vals[0][next[0]], vals[1][next[1]], vals[2][next[2]],
                        cand, delta, accept ? "accept" : "reject", best_score);
        }
    }

    return BlockConfig{ vals[0][best[0]], vals[1][best[1]], vals[2][best[2]], best_score };
}
