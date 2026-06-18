#pragma once

#include <cstddef>
#include <cmath>
#include <immintrin.h>

constexpr size_t MR512_8x16 = 8;
constexpr size_t NR512_8x16 = 16;

static_assert(MR512_8x16 == 8 && NR512_8x16 == 16,
              "simd512kernel_8x16 hard-codes 16 cols (2 x __m512d); the b1 column offset uses MR512_8x16, so it must be 8");

void matmat_blocked_ijk(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K,
                        size_t BM, size_t BN, size_t BK);
void matmat_blocked_ikj(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K,
                        size_t BM, size_t BN, size_t BK);
void matmat_blocked_jik(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K,
                        size_t BM, size_t BN, size_t BK);
void matmat_blocked_jki(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K,
                        size_t BM, size_t BN, size_t BK);
void matmat_blocked_kij(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K,
                        size_t BM, size_t BN, size_t BK);
void matmat_blocked_kji(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K,
                        size_t BM, size_t BN, size_t BK);

void inner_simd512_8x16(const double* A, const double* B, double* C,
                         size_t M, size_t N, size_t K);

// ------------------- For reference -----------------------
void matmat_ikj(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);