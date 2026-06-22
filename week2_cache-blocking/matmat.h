#pragma once

#include <cstddef>
#include <cmath>

void matmat_blocked(const double* A, const double* B, double* C,
                    size_t M, size_t N, size_t K,
                    size_t BM, size_t BN, size_t BK);

void matmat_inner_ikj(const double* A, const double* B, double* C, 
                      size_t ii, size_t jj, size_t kk,
                      size_t bm, size_t bn, size_t bk,
                      size_t M, size_t N, size_t K);

// ------------------- For reference -----------------------
void matmat_ikj(const double* A, const double* B, double* C, size_t M, size_t N, size_t K);