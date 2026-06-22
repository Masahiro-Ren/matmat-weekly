#include "matmat.h"

void matmat_blocked(const double* A, const double* B, double* C,
                    size_t BM, size_t BN, size_t BK,
                    size_t M, size_t N, size_t K)
{
    for(size_t ii = 0; ii < M; ii += BM)
    {
        for(size_t jj = 0; jj < N; jj += BN)
        {
            for(size_t kk = 0; kk < K; kk += BK)
            {
                const size_t bm = std::min(BM, M - ii);
                const size_t bn = std::min(BN, N - jj);
                const size_t bk = std::min(BK, K - kk);
                matmat_inner_ikj(A, B, C, ii, jj, kk, bm, bn, bk, M, N, K);
            }
        }
    }
}


void matmat_inner_ikj(const double* A, const double* B, double* C, 
                      size_t ii, size_t jj, size_t kk,
                      size_t bm, size_t bn, size_t bk,
                      size_t M, size_t N, size_t K)
{
    for(size_t i = ii; i < ii + bm; i++)
    {
        for(size_t k = kk; k < kk + bk; k++)
        {
            const double a = A[i * K + k];
            for(size_t j = jj; j < jj + bn; j++)
            {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

// ------------------- For reference -----------------------
void matmat_ikj(const double* A, const double* B, double* C, size_t M, size_t N, size_t K)
{
    for(size_t i = 0; i < M; i++)
    {
        for(size_t k = 0; k < K; k++)
        {
            double a = A[i * K + k];
            for(size_t j = 0; j < N; j++)
            {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}