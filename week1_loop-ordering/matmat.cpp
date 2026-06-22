#include "matmat.h"

void matmat_ijk(const double* A, const double* B, double* C, size_t M, size_t N, size_t K)
{
    for(size_t i = 0; i < M; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            double prod = 0.0;
            for(size_t k = 0; k < K; k++)
            {
                prod += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = prod;
        }
    }
}

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

void matmat_jik(const double* A, const double* B, double* C, size_t M, size_t N, size_t K)
{
    for(size_t j = 0; j < N; j++)
    {
        for(size_t i = 0; i < M; i++)
        {
            double prod = 0.0;
            for(size_t k = 0; k < K; k++)
            {
                prod += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = prod;
        }
    }
}

void matmat_jki(const double* A, const double* B, double* C, size_t M, size_t N, size_t K)
{
    for(size_t j = 0; j < N; j++)
    {
        for(size_t k = 0; k < K; k++)
        {
            double b = B[k * N  + j];
            for(size_t i = 0; i < M; i++)
            {
                C[i * N + j] += A[i * K + k] * b;
            }
        }
    }
}

void matmat_kij(const double* A, const double* B, double* C, size_t M, size_t N, size_t K)
{
    for(size_t k = 0; k < K; k++)
    {
        for(size_t i = 0; i < M; i++)
        {
            double a = A[i * K + k];
            for(size_t j = 0; j < N; j++)
            {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void matmat_kji(const double* A, const double* B, double* C, size_t M, size_t N, size_t K)
{
    for(size_t k = 0; k < K; k++)
    {
        for(size_t j = 0; j < N; j++)
        {
            double b = B[k * N + j];
            for(size_t i = 0; i < M; i++)
            {
                C[i * N + j] += A[i * K + k] * b;
            }
        }
    }
}