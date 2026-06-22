#pragma once

#include <cstddef>
#include <cmath>
#include <immintrin.h>

// Kernel size for AVX2 (256bit)
constexpr size_t MR256_4x4 = 4;
constexpr size_t NR256_4x4 = 4;

constexpr size_t MR256_4x8 = 4;
constexpr size_t NR256_4x8 = 8;

// Kernel size for AVX512 (512bit)
constexpr size_t MR512_8x8 = 8;
constexpr size_t NR512_8x8 = 8;

constexpr size_t MR512_8x16 = 8;
constexpr size_t NR512_8x16 = 16;

// The kernels are hand-unrolled for these exact tile shapes; changing a
// constant without rewriting the matching kernel would be a silent bug.
static_assert(MR256_4x4 == 4 && NR256_4x4 == 4,
              "simd256kernel_4x4 hard-codes a 4x4 tile (4 accumulators c0..c3, one __m256d = 4 cols)");
static_assert(MR256_4x8 == 4 && NR256_4x8 == 8,
              "simd256kernel_4x8 hard-codes 8 cols (2 x __m256d); the b1 column offset uses MR256_4x8, so it must be 4");
static_assert(MR512_8x8 == 8 && NR512_8x8 == 8,
              "simd512kernel_8x8 hard-codes an 8x8 tile (8 accumulators c0..c7, one __m512d = 8 cols)");
static_assert(MR512_8x16 == 8 && NR512_8x16 == 16,
              "simd512kernel_8x16 hard-codes 16 cols (2 x __m512d); the b1 column offset uses MR512_8x16, so it must be 8");

void matmat_simd256_4x4(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K);
void matmat_simd256_4x8(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K);

void matmat_simd512_8x8(const double* A, const double* B, double* C,
                        size_t M, size_t N, size_t K);
void matmat_simd512_8x16(const double* A, const double* B, double* C,
                         size_t M, size_t N, size_t K);

// ------------------- For reference -----------------------
void matmat_ikj(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);