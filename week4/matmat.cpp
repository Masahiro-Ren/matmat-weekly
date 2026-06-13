#include "matmat.h"

void simd256kernel_4x4(const double* A, const double* B, double* C,
                       size_t ic, size_t jc, size_t N, size_t K)
{
    // Define Accumulators
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    // Computations
    for(size_t k = 0; k < K; k++)
    {
        // Load 4 cols of B at row k
        __m256d b = _mm256_loadu_pd(&B[k * N + jc]);
        // Load and broadcasting values of A[ic + ir][k]
        __m256d a0 = _mm256_broadcast_sd(&A[(ic + 0) * K + k]);
        __m256d a1 = _mm256_broadcast_sd(&A[(ic + 1) * K + k]);
        __m256d a2 = _mm256_broadcast_sd(&A[(ic + 2) * K + k]);
        __m256d a3 = _mm256_broadcast_sd(&A[(ic + 3) * K + k]);
        // Kernel computations
        c0 = _mm256_fmadd_pd(a0, b, c0);
        c1 = _mm256_fmadd_pd(a1, b, c1);
        c2 = _mm256_fmadd_pd(a2, b, c2);
        c3 = _mm256_fmadd_pd(a3, b, c3);
    }

    // Store Results
    _mm256_storeu_pd(&C[(ic + 0) * N + jc], c0);
    _mm256_storeu_pd(&C[(ic + 1) * N + jc], c1);
    _mm256_storeu_pd(&C[(ic + 2) * N + jc], c2);
    _mm256_storeu_pd(&C[(ic + 3) * N + jc], c3);

}

void simd512kernel_8x16(const double* A, const double* B, double* C,
                        size_t ic, size_t jc, size_t N, size_t K)
{
    // Define Accumulators
    __m512d c[MR_512][2];
    for(size_t ir = 0; ir < MR_512; ir++)
    {
        c[ir][0] = _mm512_setzero_pd();
        c[ir][1] = _mm512_setzero_pd();
    }

    // Computations
    for(size_t k = 0; k < K; k++)
    {
        // Load 2 sets (16 cols) of B at row k
        __m512d b0 = _mm512_loadu_pd(&B[k * N + jc]);
        __m512d b1 = _mm512_loadu_pd(&B[k * N + jc + MR_512]);

        for(size_t ir = 0; ir < MR_512; ir++)
        {
            // Load and broad casting values of A[ic + ir][k]
            __m512d a = _mm512_set1_pd(A[(ic + ir) * K + k]);
            // Kernel Computations
            c[ir][0] = _mm512_fmadd_pd(a, b0, c[ir][0]);
            c[ir][1] = _mm512_fmadd_pd(a, b1, c[ir][1]);
        }
    }

    // Store reuslts
    for(size_t ir = 0; ir < MR_512; ir++)
    {
        _mm512_storeu_pd(&C[(ic + ir) * N + jc], c[ir][0]);
        _mm512_storeu_pd(&C[(ic + ir) * N + (jc + 8)], c[ir][1]);
    }
}

void matmat_simd256(const double* A, const double* B, double* C,
                    size_t M, size_t N, size_t K)
{
    for(size_t i = 0; i < M; i += MR_256)
    {
        for(size_t j = 0; j < N; j += NR_256)
        {
            simd256kernel_4x4(A, B, C, i, j, N, K);
        }
    }
}

void matmat_simd512(const double* A, const double* B, double* C,
                    size_t M, size_t N, size_t K)
{
    for(size_t i = 0; i < M; i += MR_512)
    {
        for(size_t j = 0; j < N; j += NR_512)
        {
            simd512kernel_8x16(A, B, C, i, j, N, K);
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