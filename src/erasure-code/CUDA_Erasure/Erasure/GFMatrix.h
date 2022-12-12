#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef struct
{
    uint8_t **data;
    size_t m;
    size_t n;
} GFMatrix;

uint8_t gfMultiply[256][256];
uint8_t gfInverse[256];

void initGF()
{
    FILE *gfMultFile;
    if(!(gfMultFile = fopen("gfmul.bin", "rb")))
    {
        printf("Can't find galois field mult file\n");
        exit(0);
    }
    FILE *gfInvFile;
    if(!(gfInvFile = fopen("gfinv.bin", "rb")))
    {
        printf("Can't find galois field inverse file\n");
        exit(0);
    }
    fread(gfMultiply, sizeof(uint8_t), 256*256, gfMultFile);
    fread(gfInverse, sizeof(uint8_t), 256, gfInvFile);
    fclose(gfMultFile);
    fclose(gfInvFile);
}

GFMatrix matrixAlloc(size_t m, size_t n)
{
    GFMatrix outMatrix;
    outMatrix.m = m;
    outMatrix.n = n;
    outMatrix.data = (uint8_t **) malloc(m * sizeof(uint8_t *));
    for(size_t i = 0; i < m; i++)
    {
        outMatrix.data[i] = (uint8_t *) malloc(n * sizeof(uint8_t));
    }
    return outMatrix;
}

void freeMatrix(GFMatrix *mat)
{
    for(size_t i = 0; i<mat->m; i++)
    {
        free(mat->data[i]);
    }
    free(mat->data);
}

void insertAt(size_t i, size_t j, int val, GFMatrix *m)
{
    if(i < m->m || j < m->n)
        m->data[i][j] = val;
}

int GFMatrixGetAt(size_t i, size_t j, GFMatrix *m)
{
    return m->data[i][j];
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
GFMatrix addGFMatrix(GFMatrix *a, GFMatrix *b)
{
    GFMatrix zeroMatrix;
    if(a->m != b->m || a->n != b->n)
    {
        return zeroMatrix;
    }
    GFMatrix sum;
    sum.m = a->m;
    sum.n = a->n;
    for(size_t i = 0; i<a->m; i++)
    {
        for(size_t j = 0; j<a->n; j++)
        {
            insertAt(i, j, GFMatrixGetAt(i, j, a) ^ GFMatrixGetAt(i, j, b), &sum);
        }
    }
    return sum;
}

uint8_t GFVectorDot(uint8_t *v1, uint8_t *v2, size_t n)
{
    int sum = 0;
    for(size_t i = 0; i<n; i++)
    {
        sum = sum ^ gfMultiply[v1[i]][v2[i]];
    }

    return sum;
}

void GFMatrixVectorProd(GFMatrix *m, uint8_t *inVec, uint8_t *outVec)
{
    for(int i = 0; i < m->m; i++)
    {
        outVec[i] = GFVectorDot(m->data[i], inVec, m->n);
    }
}

void GFVectorScalarMultiply(uint8_t *inVec, uint8_t scalar, uint8_t *outVec, size_t n)
{
    for(size_t i = 0; i<n; i++)
    {
        outVec[i] = gfMultiply[inVec[i]][scalar];
    }
}

void GFVectorAdd(uint8_t *a, uint8_t *b, uint8_t *out, size_t n)
{
    for(size_t i = 0; i<n; i++)
    {
        out[i] = a[i] ^ b[i];
    }
}

void GFGaussianElimination(GFMatrix *mat)
{
    //uint8_t *scale = (uint8_t *)malloc(mat->n * sizeof(uint8_t));
    uint8_t scale[mat->n];
    for(size_t i = 0; i<mat->m; i++)
    {
        size_t j = 0;
        for(; j<i; j++)
        {
            if(mat->data[i][j] == 0) continue;
            GFVectorScalarMultiply(mat->data[j], mat->data[i][j], scale, mat->n);
            GFVectorAdd(mat->data[i], scale, mat->data[i], mat->n);
        }
        if(mat->data[i][j] == 1) continue;
        GFVectorScalarMultiply(mat->data[i], gfInverse[mat->data[i][j]], mat->data[i], mat->n);
    }

    for(size_t i = (mat->n)-1; i>0; i--)
    {
        for(size_t j = (mat->m)-1; j>i; j--)
        {
            if(mat->data[i][j] == 0) continue;
            GFVectorScalarMultiply(mat->data[j], mat->data[i][j], scale, mat->n);
            GFVectorAdd(mat->data[i], scale, mat->data[i], mat->n);
        }
    }

    for(size_t j = (mat->m)-1; j>0; j--)
    {
        if(mat->data[0][j] == 0) continue;
        GFVectorScalarMultiply(mat->data[j], mat->data[0][j], scale, mat->n);
        GFVectorAdd(mat->data[0], scale, mat->data[0], mat->n);
    }
    //free(scale);
}