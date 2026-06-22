#include "matmat.h"

void simd512kernel_8x16(const double* A, const double* B, double* C,
                        size_t ic, size_t jc, 
                        size_t kk, size_t bk,
                        size_t N, size_t K)
{
    // Define Accumulators
    __m512d c[MR512_8x16][2];
    for(size_t ir = 0; ir < MR512_8x16; ir++)
    {
        c[ir][0] = _mm512_loadu_pd(&C[(ic + ir) * N + jc]);
        c[ir][1] = _mm512_loadu_pd(&C[(ic + ir) * N + jc + 8]);
    }

    // Computations
    for(size_t k = kk; k < kk + bk; k++)
    {
        // Load 2 sets (16 cols) of B at row k
        __m512d b0 = _mm512_loadu_pd(&B[k * N + jc]);
        __m512d b1 = _mm512_loadu_pd(&B[k * N + jc + MR512_8x16]);

        for(size_t ir = 0; ir < MR512_8x16; ir++)
        {
            // Load and broadcasting values of A[ic + ir][k]
            __m512d a = _mm512_set1_pd(A[(ic + ir) * K + k]);
            // Kernel Computations
            c[ir][0] = _mm512_fmadd_pd(a, b0, c[ir][0]);
            c[ir][1] = _mm512_fmadd_pd(a, b1, c[ir][1]);
        }
    }

    // Store reuslts
    for(size_t ir = 0; ir < MR512_8x16; ir++)
    {
        _mm512_storeu_pd(&C[(ic + ir) * N + jc], c[ir][0]);
        _mm512_storeu_pd(&C[(ic + ir) * N + (jc + 8)], c[ir][1]);
    }
}


void inner_simd512_8x16(const double* A, const double* B, double* C,
                        size_t ii, size_t jj, size_t kk,
                        size_t bm, size_t bn, size_t bk,
                        size_t M, size_t N, size_t K)
{
    for(size_t i = ii; i < ii + bm; i += MR512_8x16)
    {
        for(size_t j = jj; j < jj + bn; j += NR512_8x16)
        {
            simd512kernel_8x16(A, B, C, i, j, kk, bk, N, K);
        }
    }
}

void matmat_blocked_ijk(const double* A, const double* B, double* C,
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
                inner_simd512_8x16(A, B, C, ii, jj, kk, bm, bn, bk, M, N, K); 
            }
        }
    }
}

void matmat_blocked_ikj(const double* A, const double* B, double* C,
                        size_t BM, size_t BN, size_t BK,
                        size_t M, size_t N, size_t K)
{
    for(size_t ii = 0; ii < M; ii += BM)
    {
        for(size_t kk = 0; kk < K; kk += BK)
        {
            for(size_t jj = 0; jj < N; jj += BN)
            {
                const size_t bm = std::min(BM, M - ii);
                const size_t bn = std::min(BN, N - jj);
                const size_t bk = std::min(BK, K - kk);
                inner_simd512_8x16(A, B, C, ii, jj, kk, bm, bn, bk, M, N, K); 
            }
        }
    }
}

void matmat_blocked_jik(const double* A, const double* B, double* C,
                        size_t BM, size_t BN, size_t BK,
                        size_t M, size_t N, size_t K)
{
    for(size_t jj = 0; jj < N; jj += BN)
    {
        for(size_t ii = 0; ii < M; ii += BM)
        {
            for(size_t kk = 0; kk < K; kk += BK)
            {
                const size_t bm = std::min(BM, M - ii);
                const size_t bn = std::min(BN, N - jj);
                const size_t bk = std::min(BK, K - kk);
                inner_simd512_8x16(A, B, C, ii, jj, kk, bm, bn, bk, M, N, K); 
            }
        }
    }
}

void matmat_blocked_jki(const double* A, const double* B, double* C,
                        size_t BM, size_t BN, size_t BK,
                        size_t M, size_t N, size_t K)
{
    for(size_t jj = 0; jj < N; jj += BN)
    {
        for(size_t kk = 0; kk < K; kk += BK)
        {
            for(size_t ii = 0; ii < M; ii += BM)
            {
                const size_t bm = std::min(BM, M - ii);
                const size_t bn = std::min(BN, N - jj);
                const size_t bk = std::min(BK, K - kk);
                inner_simd512_8x16(A, B, C, ii, jj, kk, bm, bn, bk, M, N, K); 
            }
        }
    }
}

void matmat_blocked_kij(const double* A, const double* B, double* C,
                        size_t BM, size_t BN, size_t BK,
                        size_t M, size_t N, size_t K)
{
    for(size_t kk = 0; kk < K; kk += BK)
    {
        for(size_t ii = 0; ii < M; ii += BM)
        {
            for(size_t jj = 0; jj < N; jj += BN)
            {
                const size_t bm = std::min(BM, M - ii);
                const size_t bn = std::min(BN, N - jj);
                const size_t bk = std::min(BK, K - kk);
                inner_simd512_8x16(A, B, C, ii, jj, kk, bm, bn, bk, M, N, K); 
            }
        }
    }
}

void matmat_blocked_kji(const double* A, const double* B, double* C,
                        size_t BM, size_t BN, size_t BK,
                        size_t M, size_t N, size_t K)
{
    for(size_t kk = 0; kk < K; kk += BK)
    {
        for(size_t jj = 0; jj < N; jj += BN)
        {
            for(size_t ii = 0; ii < M; ii += BM)
            {
                const size_t bm = std::min(BM, M - ii);
                const size_t bn = std::min(BN, N - jj);
                const size_t bk = std::min(BK, K - kk);
                inner_simd512_8x16(A, B, C, ii, jj, kk, bm, bn, bk, M, N, K); 
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