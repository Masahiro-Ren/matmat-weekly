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
    for(size_t k = 0; k < bk; k++)
    {
        // Load 2 sets (16 cols) of Bpack at row k
        __m512d b0 = _mm512_loadu_pd(&B[k * NR512_8x16]);
        __m512d b1 = _mm512_loadu_pd(&B[k * NR512_8x16 + MR512_8x16]);

        for(size_t ir = 0; ir < MR512_8x16; ir++)
        {
            // Load and broadcasting values of Apack
            __m512d a = _mm512_set1_pd(A[k * MR512_8x16 + ir]);
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


void inner_simd512_8x16(const double* Apack, const double* Bpack, double* C,
                        size_t ii, size_t jj, size_t kk,
                        size_t bm, size_t bn, size_t bk,
                        size_t M, size_t N, size_t K)
{
    for(size_t i = 0; i < bm; i += MR512_8x16)
    {
        for(size_t j = 0; j < bn; j += NR512_8x16)
        {
            const size_t ic = ii + i;
            const size_t jc = jj + j;
            simd512kernel_8x16(Apack + i * bk, Bpack + j * bk, C, ic, jc, kk, bk, N, K);
        }
    }
}

void pack_A(const double* A, double* Apack, 
            size_t ii, size_t kk, 
            size_t bm, size_t bk, 
            size_t M, size_t K)
{
    for(size_t i = 0; i < bm; i += MR512_8x16)
    {
        for(size_t k = 0; k < bk; k++)
        {
            for(size_t ir = 0; ir < MR512_8x16; ir++)
            {
                Apack[i * bk + k * MR512_8x16 + ir] = A[(ii + i + ir) * K + (kk + k)];
            }
        }
    }
}

void pack_B(const double* B, double* Bpack, 
            size_t kk, size_t jj, 
            size_t bk, size_t bn, 
            size_t K, size_t N)
{
    for(size_t j = 0; j < bn; j += NR512_8x16)
    {
        for(size_t k = 0; k < bk; k++)
        {
            for(size_t jc = 0; jc < NR512_8x16; jc++)
            {
                Bpack[j * bk + k * NR512_8x16 + jc] = B[(kk + k) * N + (jj + j + jc)];
            }
        }
    }
}


void matmat_blocked_ikj(const double* A, const double* B, double* C,
                        size_t BM, size_t BN, size_t BK,
                        size_t M, size_t N, size_t K)
{
    #pragma omp parallel
    {
        // double* Apack = static_cast<double*>(std::aligned_alloc(64, BM * BK * sizeof(double)));
        // double* Bpack = static_cast<double*>(std::aligned_alloc(64, BK * BN * sizeof(double)));
        static thread_local double* Apack = nullptr;  static thread_local size_t Acap = 0;
        static thread_local double* Bpack = nullptr;  static thread_local size_t Bcap = 0;
        if (BM*BK > Acap) { std::free(Apack); Apack = (double*)std::aligned_alloc(64, BM*BK*sizeof(double)); Acap = BM*BK; }
        if (BK*BN > Bcap) { std::free(Bpack); Bpack = (double*)std::aligned_alloc(64, BK*BN*sizeof(double)); Bcap = BK*BN; }

        #pragma omp for schedule(static)
        for(size_t ii = 0; ii < M; ii += BM)
        {
            const size_t bm = std::min(BM, M - ii);
            for(size_t kk = 0; kk < K; kk += BK)
            {
                const size_t bk = std::min(BK, K - kk);
                pack_A(A, Apack, ii, kk, bm, bk, M, K);

                for(size_t jj = 0; jj < N; jj += BN)
                {
                    const size_t bn = std::min(BN, N - jj);
                    pack_B(B, Bpack, kk, jj, bk, bn, K, N);

                    inner_simd512_8x16(Apack, Bpack, C, ii, jj, kk, bm, bn, bk, M, N, K); 
                }
            }
        }

        // std::free(Apack);
        // std::free(Bpack);
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