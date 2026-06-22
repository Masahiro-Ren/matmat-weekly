#pragma once

#include <cstddef>

void matmat_ijk(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);

void matmat_ikj(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);

void matmat_jik(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);

void matmat_jki(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);

void matmat_kij(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);

void matmat_kji(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);