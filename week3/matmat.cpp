#include "matmat.h"

void microkernel_4x4(const double* A, const double* B, double* C,
                     size_t ic, size_t jc, size_t N, size_t K)
{
    // Define an accumulator
    double acc[MR][NR] = {};

    // Computations
    for(size_t k = 0; k < K; k++)
    {
        // Load MR A values
        double as[MR];
        #pragma GCC unroll (MR)
        for(size_t ir = 0; ir < MR; ir++)
            as[ir] = A[(ic + ir) * K + k];
        
        // Load NR B values
        const double* bs = &B[k * N + jc];

        // Kernel Computations
        for(size_t jr = 0; jr < NR; jr++)
        {
            const double b = bs[jr];
            #pragma GCC unroll (MR)
            for(size_t ir = 0; ir < MR; ir++)
            {
                acc[ir][jr] += as[ir] * b;
            }
        }
    }

    // Write back to C
    for(size_t ir = 0; ir < MR; ir++)
    {
        for(size_t jr = 0; jr < NR; jr++)
        {
            C[(ic + ir) * N + (jc +jr)] = acc[ir][jr];
        }
    }
}

void microkernel_4x3(const double* A, const double* B, double* C,
                     size_t ic, size_t jc, size_t N, size_t K)
{
    // Define an accumulator
    double acc[MR][NRR] = {};

    const size_t nr = std::min(NRR, N - jc);

    // Computations
    for(size_t k = 0; k < K; k++)
    {
        // Load MR A values
        double as[MR];
        #pragma GCC unroll (MR)
        for(size_t ir = 0; ir < MR; ir++)
            as[ir] = A[(ic + ir) * K + k];
        
        // Load NRR B values
        const double* bs = &B[k * N + jc];

        // Kernel Computations
        for(size_t jr = 0; jr < nr; jr++)
        {
            const double b = bs[jr];
            #pragma GCC unroll (MR)
            for(size_t ir = 0; ir < MR; ir++)
            {
                acc[ir][jr] += as[ir] * b;
            }
        }
    }

    // Write back to C
    for(size_t ir = 0; ir < MR; ir++)
    {
        for(size_t jr = 0; jr < nr; jr++)
        {
            C[(ic + ir) * N + (jc +jr)] = acc[ir][jr];
        }
    }
}

void matmat_micro4x4(const double* A, const double* B, double* C,
                     size_t M, size_t N, size_t K)
{
    for(size_t i = 0; i < M; i += MR)
    {
        for(size_t j = 0; j < N; j += NR)
        {
            microkernel_4x4(A, B, C, i, j, N, K);
        }
    }
}

void matmat_micro4x3(const double* A, const double* B, double* C,
                     size_t M, size_t N, size_t K)
{

    for(size_t i = 0; i < M; i += MR)
    {
        for(size_t j = 0; j < N; j += NRR)
        {
            microkernel_4x3(A, B, C, i, j, N, K);
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