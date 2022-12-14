#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <cuda.h>

typedef struct
{
    //used as a 1d representation of a 2d array
    uint8_t *data;
    size_t m;
    size_t n;
} GFMatrix;

__device__ uint8_t gfMultiply[256*256];
__device__ uint8_t gfInverse[256];
uint8_t host_gfMultiply[256*256];
uint8_t host_gfInverse[256];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void initGF()
{
    FILE *gfMultFile;

    if(!(gfMultFile = fopen("/home/ben/gitroot/fall2022-bathompson/build/gfmul.bin", "rb")))
    {
        printf("Can't find galois field mult file\n");
        exit(0);
    }
    FILE *gfInvFile;
    if(!(gfInvFile = fopen("/home/ben/gitroot/fall2022-bathompson/build/gfinv.bin", "rb")))
    {
        printf("Can't find galois field inverse file\n");
        exit(0);
    }
    fread(host_gfMultiply, sizeof(uint8_t), 256*256, gfMultFile);
    fread(host_gfInverse, sizeof(uint8_t), 256, gfInvFile);
    fclose(gfMultFile);
    fclose(gfInvFile);
    uint8_t *h_mul;
    uint8_t *h_inv;
    gpuErrchk(cudaMalloc(&h_mul, 256*256));
    gpuErrchk(cudaMalloc(&h_inv, 256));
    gpuErrchk(cudaMemcpy(h_mul, host_gfMultiply, 256*256*sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_inv, host_gfInverse, 256*sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(h_mul));
    gpuErrchk(cudaFree(h_inv));
}

__global__ void cpyToDevice(uint8_t *h_mul, uint8_t *h_inv)
{
    //cudaMemcpy(gfMultiply, h_mul, 256*256*sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    //cudaMemcpy(gfInverse, h_inv, 256*sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    memcpy(gfMultiply, h_mul, 256*256*sizeof(uint8_t));
    memcpy(gfInverse, h_inv, 256*sizeof(uint8_t));
}

__device__ GFMatrix matrixAlloc(size_t m, size_t n)
{
    GFMatrix outMatrix;
    outMatrix.m = m;
    outMatrix.n = n;
    outMatrix.data = (uint8_t *) malloc(m*n*sizeof(uint8_t));
    //cudaMalloc(&outMatrix.data, n*m*sizeof(uint8_t));
    return outMatrix;
}

__device__ __host__ void insertAt(size_t i, size_t j, int val, GFMatrix *m)
{
    if(i < m->m || j < m->n)
        m->data[i*(m->n)+j] = val;
}

__device__ __host__ int GFMatrixGetAt(size_t i, size_t j, GFMatrix *m)
{
    return m->data[i*m->n + j];
}

/**
 * @brief 
 * 
 * @param a 
 * @param b 
 * @param sum allocated matrix to store the sum.
 * @return true 
 * @return false 
 */
// GFMatrix addGFMatrix(GFMatrix *a, GFMatrix *b)
// {
//     GFMatrix zeroMatrix;
//     zeroMatrix.data = NULL;
//     zeroMatrix.m = 0;
//     zeroMatrix.n = 0;
//     if(a->m != b->m || a->n != b->n)
//     {
//         return zeroMatrix;
//     }
//     GFMatrix sum;
//     sum.m = a->m;
//     sum.n = a->n;
//     for(size_t i = 0; i<a->m; i++)
//     {
//         for(size_t j = 0; j<a->n; j++)
//         {
//             insertAt(i, j, GFMatrixGetAt(i, j, a) ^ GFMatrixGetAt(i, j, b), &sum);
//         }
//     }
//     return sum;
// }

__device__ uint8_t GFVectorDot(uint8_t *v1, uint8_t *v2, size_t n)
{
    int sum = 0;
    for(size_t i = 0; i<n; i++)
    {
        uint8_t val = gfMultiply[v1[i]*256+v2[i]];
        sum = sum ^ val;
    }

    return sum;
}

__device__ void GFMatrixVectorProd(GFMatrix *m, uint8_t *inVec, uint8_t *outVec)
{
    for(int i = 0; i < m->m; i++)
    {
        outVec[i] = GFVectorDot(&(m->data[i*(m->n)]), inVec, m->n);
    }
}

__device__ void GFVectorScalarMultiply(uint8_t *inVec, uint8_t scalar, uint8_t *outVec, size_t n)
{
    for(size_t i = 0; i<n; i++)
    {
        outVec[i] = gfMultiply[inVec[i]*256 + scalar];
    }
}

__device__ void GFVectorAdd(uint8_t *a, uint8_t *b, uint8_t *out, size_t n)
{
    for(size_t i = 0; i<n; i++)
    {
        out[i] = a[i] ^ b[i];
    }
}

__device__ void GFGaussianElimination(GFMatrix *mat)
{
    uint8_t *scale = (uint8_t *)malloc(mat->n * sizeof(uint8_t));
    
    for(size_t i = 0; i<mat->m; i++)
    {
        size_t j = 0;
        for(; j<i; j++)
        {
            if(mat->data[i*(mat->n)+j] == 0) continue;
            GFVectorScalarMultiply((mat->data) + (j*mat->n), mat->data[i*mat->n+j], scale, mat->n);
            GFVectorAdd((mat->data)+mat->n*i, scale, (mat->data)+mat->n*i, mat->n);
        }
        if(mat->data[i*(mat->n) + j] == 1) continue;
        GFVectorScalarMultiply((mat->data)+(mat->n)*i, gfInverse[mat->data[i*(mat->n)+j]], (mat->data)+(mat->n)*i, mat->n);
    }

    for(size_t i = (mat->n)-1; i>0; i--)
    {
        for(size_t j = (mat->m)-1; j>i; j--)
        {
            if(mat->data[i*(mat->n)+j] == 0) continue;
            GFVectorScalarMultiply((mat->data) + (j*mat->n), mat->data[i*mat->n+j], scale, mat->n);
            GFVectorAdd((mat->data)+(mat->n)*i, scale, (mat->data)+(mat->n)*i, mat->n);
        }
    }

    for(size_t j = (mat->m)-1; j>0; j--)
        {
            if(mat->data[j] == 0) continue;
            GFVectorScalarMultiply((mat->data) + (j*mat->n), mat->data[j], scale, mat->n);
            GFVectorAdd(mat->data, scale, mat->data, mat->n);
        }
        free(scale);
}