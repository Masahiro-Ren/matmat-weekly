#pragma once

#include <cstddef>
#include <cmath>
#include <immintrin.h>

// Kernel size for AVX2 (256bit)
constexpr size_t MR_256 = 4;
constexpr size_t NR_256 = 4;

// Kernel size for AVX512 (512bit)
constexpr size_t MR_512 = 8;
constexpr size_t NR_512 = 16;

// The kernels are hand-unrolled for these exact tile shapes; changing a
// constant without rewriting the matching kernel would be a silent bug.
static_assert(MR_256 == 4 && NR_256 == 4,
              "simd256kernel is hard-coded for a 4x4 (MR_256 x NR_256) tile "
              "(4 accumulators c0..c3, one __m256d = 4 doubles per row)");
static_assert(NR_512 == 16,
              "simd512kernel is hard-coded for 16 columns (2 x __m512d of 8 doubles)");

void matmat_simd256(const double* A, const double* B, double* C,
                    size_t M, size_t N, size_t K);

void matmat_simd512(const double* A, const double* B, double* C,
                    size_t M, size_t N, size_t K);

// ------------------- For reference -----------------------
void matmat_ikj(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);