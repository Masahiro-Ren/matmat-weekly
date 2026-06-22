#pragma once

#include "bench.h"

struct BlockConfig {
    size_t BM;
    size_t BN;
    size_t BK;
    double score;
};

struct SAParams
{
    double T0 = 200.0;
    double alpha = 0.9;
    double Tmin = 1.0;
    int per_T = 6;
    unsigned seed = 12345;
};


BlockConfig autotune(const double* A, const double* B, double* C,
                     size_t M, size_t N, size_t K,
                     int warmup, int rounds, SAParams param = {});