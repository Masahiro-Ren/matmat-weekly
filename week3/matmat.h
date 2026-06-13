#pragma once

#include <cstddef>
#include <cmath>

constexpr size_t MR = 4;
constexpr size_t NR = 4;
constexpr size_t NRR = 3;

// --------------------- 4x4 micro kernel ----------------------
void matmat_micro4x4(const double* A, const double* B, double* C,
                     size_t M, size_t N, size_t K);

// --------------------- 4x3 micro kernel ----------------------
void matmat_micro4x3(const double* A, const double* B, double* C,
                     size_t M, size_t N, size_t K);

// ------------------- For reference -----------------------
void matmat_ikj(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);