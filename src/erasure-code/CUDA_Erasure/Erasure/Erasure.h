#include "GFMatrix.h"

GFMatrix allocIdMatrix(size_t n)
{
    GFMatrix id = matrixAlloc(n, n);
    for(size_t i = 0; i<n; i++)
    {
        for(size_t j = 0; j<n; j++)
        {
            if(i == j)
                id.data[i][j] = 1;
            else
                id.data[i][j] = 0;
        }
    }
    return id;
}

GFMatrix allocVandMatrix(size_t n, size_t k)
{
    GFMatrix vand;
    vand = matrixAlloc(k, n);
    for(size_t i = 0; i<vand.n; i++)
    {
        insertAt(0, i, 1, &vand);
    }
    for(size_t i = 1; i<vand.m; i++)
    {
        for(size_t j = 0; j<vand.n; j++)
        {
            insertAt(i, j, gfMultiply[GFMatrixGetAt(i-1, j, &vand)][j+1], &vand);
        }
    }
    return vand;
}

GFMatrix allocAugmentedMatrix(GFMatrix *m, uint8_t *vec)
{
    GFMatrix aug = matrixAlloc(m->m, (m->n)+1);
    for(size_t i = 0; i<m->m; i++)
    {
        memcpy(aug.data[i], m->data[i], m->n);
    }
    for(size_t i = 0; i<m->m; i++)
    {
        aug.data[i][m->n] = vec[i];
    }
    return aug;
}

void encodeData(uint8_t **data, size_t n, size_t k, size_t chunkLength, uint8_t **codes)
{
    initGF();
    GFMatrix vand = allocVandMatrix(n, k);
    uint8_t *curCode = (uint8_t *)malloc(k*sizeof(uint8_t));
    uint8_t *curTuple = (uint8_t *)malloc(n*sizeof(uint8_t));
    for(size_t i = 0; i<chunkLength; i++)
    {
        for(size_t j = 0; j<n; j++)
        {
            curTuple[j] = data[j][i];
        }
        GFMatrixVectorProd(&vand, curTuple, curCode);
        for(size_t j = 0; j<k; j++)
        {
            codes[j][i] = curCode[j];
        }
    }
    free(curCode);
    free(curTuple);
    freeMatrix(&vand);
}

void decodeData(uint8_t **inData, size_t *erasures, size_t *codesToUse, size_t numErasures, size_t n, size_t k, size_t chunkLength, uint8_t **outData)
{
    initGF();
    GFMatrix codeMatrix = allocIdMatrix(n);
    GFMatrix vand = allocVandMatrix(n, k);
    for(size_t i = 0; i<numErasures; i++)
    {
        if(erasures[i]>n) continue;
        memcpy(codeMatrix.data[erasures[i]], vand.data[codesToUse[i]], n);
    }

    uint8_t curTuple[n];

    //uint8_t **curTuple = (uint8_t **)malloc(omp_get_max_threads() * sizeof(uint8_t *));
    // for(int i = 0; i<omp_get_max_threads(); i++)
    // {
    //     curTuple[i] = (uint8_t *)malloc(n*sizeof(uint8_t));
    // }
    for(size_t i = 0; i<chunkLength; i++)
    {
        for(size_t j = 0; j<n; j++)
        {
            curTuple[j] = inData[j][i];
        }
        GFMatrix aug = allocAugmentedMatrix(&codeMatrix, curTuple);
        GFGaussianElimination(&aug);
        for(size_t j = 0; j<n; j++)
        {
            outData[j][i] = aug.data[j][n];
        }
        freeMatrix(&aug);
    }
    // for(int i = 0; i<omp_get_max_threads(); i++)
    // {
    //     free(curTuple[i]);
    // }
    // free(curTuple);
    freeMatrix(&codeMatrix);
    freeMatrix(&vand);
}