#include "GFMatrix.cuh"

#define MAX_THREADS_PER_BLOCK 32
#define MAX_NUM_BLOCKS 184

GFMatrix allocVandMatrixHost(size_t n, size_t k)
{
    GFMatrix vand;
    vand.n = n;
    vand.m = k;
    vand.data = (uint8_t *)malloc(n*k*sizeof(uint8_t));
    for(size_t i = 0; i<vand.n; i++)
    {
        insertAt(0, i, 1, &vand);
    }
    for(size_t i = 1; i<vand.m; i++)
    {
        for(size_t j = 0; j<vand.n; j++)
        {
            uint8_t val = host_gfMultiply[GFMatrixGetAt(i-1, j, &vand)*256 + j+1];
            insertAt(i, j, val, &vand);
        }
    }

    return vand;
}

__host__ GFMatrix *allocDecodeMatrix(size_t n, size_t k, size_t numErasures, size_t *erasures, size_t *codesToUse)
{
    GFMatrix id;
    id.m = n;
    id.n = n;
    id.data = (uint8_t *)malloc(n*n*sizeof(uint8_t));
    for(size_t i = 0; i<n; i++)
    {
        for(size_t j = 0; j<n; j++)
        {
            if(i == j)
                id.data[i*n+j] = 1;
            else
                id.data[i*n + j] = 0;
        }
    }
    GFMatrix vand = allocVandMatrixHost(n, k);
    
    for(size_t i = 0; i<numErasures; i++)
    {
        if(erasures[i]>n) continue;
        memcpy(id.data+erasures[i]*id.n, vand.data+codesToUse[i]*vand.n, n);
    }
    GFMatrix decodeMatrix;
    decodeMatrix.n = id.n;
    decodeMatrix.m = id.m;
    cudaMalloc(&decodeMatrix.data, decodeMatrix.m*decodeMatrix.n*sizeof(uint8_t));
    cudaMemcpy(decodeMatrix.data, id.data, decodeMatrix.m*decodeMatrix.n*sizeof(uint8_t), cudaMemcpyHostToDevice);
    free(id.data);
    GFMatrix *decodeMatrix_dev;
    cudaMalloc(&decodeMatrix_dev, sizeof(GFMatrix));
    cudaMemcpy(decodeMatrix_dev, &decodeMatrix, sizeof(GFMatrix), cudaMemcpyHostToDevice);
    return decodeMatrix_dev;
}

GFMatrix *allocVandMatrix(size_t n, size_t k, uint8_t **danglingCudaPtr)
{
    GFMatrix vand;
    vand.n = n;
    vand.m = k;
    vand.data = (uint8_t *)malloc(n*k*sizeof(uint8_t));
    for(size_t i = 0; i<vand.n; i++)
    {
        insertAt(0, i, 1, &vand);
    }
    for(size_t i = 1; i<vand.m; i++)
    {
        for(size_t j = 0; j<vand.n; j++)
        {
            uint8_t val = host_gfMultiply[GFMatrixGetAt(i-1, j, &vand)*256 + j+1];
            insertAt(i, j, val, &vand);
        }
    }
    GFMatrix vand_toCuda;
    vand_toCuda.n = n;
    vand_toCuda.m = k;
    cudaMalloc(&(vand_toCuda.data), n*k*sizeof(uint8_t));
    cudaMemcpy(vand_toCuda.data, vand.data, n*k*sizeof(uint8_t), cudaMemcpyHostToDevice);
    *danglingCudaPtr = vand_toCuda.data;
    GFMatrix *vand_dev;
    cudaMalloc(&vand_dev, sizeof(GFMatrix));
    cudaMemcpy(vand_dev, &vand_toCuda, sizeof(GFMatrix), cudaMemcpyHostToDevice);
    free(vand.data);

    return vand_dev;
}

__device__ GFMatrix allocAugmentedMatrix(GFMatrix *m, uint8_t *vec)
{
    GFMatrix aug = matrixAlloc(m->m, (m->n)+1);
    for(size_t i = 0; i<m->m; i++)
    {
        memcpy((aug.data) + aug.n*i, (m->data)+m->n*i, m->n);
    }
    for(size_t i = 0; i<m->m; i++)
    {
        aug.data[i*aug.n +m->n] = vec[i];
    }
    return aug;
}

// void encodeData(uint8_t **data, size_t n, size_t k, size_t chunkLength, uint8_t **codes)
// {
//     //GFMatrix vand = allocVandMatrix(n, k);
//     uint8_t *curCode = (uint8_t *)malloc(k*sizeof(uint8_t));
//     uint8_t *curTuple = (uint8_t *)malloc(n*sizeof(uint8_t));
//     for(size_t i = 0; i<chunkLength; i++)
//     {
//         for(size_t j = 0; j<n; j++)
//         {
//             curTuple[j] = data[j][i];
//         }
//         GFMatrixVectorProd(&vand, curTuple, curCode);
//         for(size_t j = 0; j<k; j++)
//         {
//             codes[j][i] = curCode[j];
//         }
//     }
//     free(curCode);
//     free(curTuple);
// }

void encodeData(uint8_t **data, size_t n, size_t k, size_t chunkLength, uint8_t **coding)
{
    size_t threadsPerBlock = min(chunkLength, (unsigned long)MAX_THREADS_PER_BLOCK);
    size_t numBlocks = (chunkLength<MAX_THREADS_PER_BLOCK) ? 1 : min( chunkLength/MAX_THREADS_PER_BLOCK + ((chunkLength %MAX_THREADS_PER_BLOCK == 0)?0:1),(unsigned long)MAX_NUM_BLOCKS);
    size_t elementsPerThread = chunkLength/(numBlocks*threadsPerBlock) +((chunkLength%numBlocks*threadsPerBlock == 0) ? 0:1);
    uint8_t *dev_data;
    uint8_t *dev_codes;
    cudaMalloc(&dev_data, n*chunkLength*sizeof(uint8_t));
    cudaMalloc(&dev_codes, k*chunkLength*sizeof(uint8_t));
    for(size_t i = 0; i<n; i++)
    {
        cudaMemcpy(dev_data+i*chunkLength, data[i], chunkLength, cudaMemcpyHostToDevice);
    }
    uint8_t *danglingPtr;
    GFMatrix *codingMatrix = allocVandMatrix(n, k, &danglingPtr);
    encodeDataKernel<<<numBlocks, threadsPerBlock>>>(dev_data, n, k, chunkLength, dev_codes, codingMatrix, elementsPerThread);
    for(size_t i = 0; i<k; i++)
    {
        cudaMemcpy(coding[i], dev_codes+i*chunkLength, chunkLength, cudaMemcpyDeviceToHost);
    }
    cudaFree(dev_data);
    cudaFree(dev_codes);
    cudaFree(danglingPtr);
    cudaFree(codingMatrix);
}
__global__ void encodeDataKernel(uint8_t *data, size_t n, size_t k, size_t chunkLength, uint8_t *codes, GFMatrix *encodingMatrix, size_t elementsPerThread)
{
    size_t ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(ix >= chunkLength) return;
    uint8_t *curCode = (uint8_t *)malloc(k*sizeof(uint8_t));
    uint8_t *curTuple = (uint8_t *)malloc(n*sizeof(uint8_t));
    
    
    for(size_t i = 0; i<elementsPerThread; i++)
    {
        size_t curIx = elementsPerThread*ix + i;
        if(curIx >= chunkLength) return;
        for(size_t j = 0; j<n; j++)
        {
            curTuple[j] = data[j*chunkLength + curIx];
        }
        GFMatrixVectorProd(encodingMatrix, curTuple, curCode);
        for(size_t j = 0; j<k; j++)
        {
            codes[j*chunkLength + curIx] = curCode[j];
        }
    }
    free(curCode);
    free(curTuple);
}

void decodeData(uint8_t **inData, size_t *erasures, size_t *codesToUse, size_t numErasures, size_t n, size_t k, size_t chunkLength, uint8_t **outData)
{
    size_t threadsPerBlock = min(chunkLength, (unsigned long)MAX_THREADS_PER_BLOCK);
    size_t numBlocks = (chunkLength<MAX_THREADS_PER_BLOCK) ? 1 : min( chunkLength/MAX_THREADS_PER_BLOCK + ((chunkLength %MAX_THREADS_PER_BLOCK == 0)?0:1),(unsigned long)MAX_NUM_BLOCKS);
    size_t elementsPerThread = chunkLength/(numBlocks*threadsPerBlock) +((chunkLength%numBlocks*threadsPerBlock == 0) ? 0:1);
    uint8_t *dev_inData;
    uint8_t *dev_outData;
    decodeDataKernel<<<numBlocks, threadsPerBlock>>>()
}

__global__ void decodeDataKernel(uint8_t *inData, size_t n, size_t k, size_t chunkLength, uint8_t *outData, GFMatrix *decodeMatrix, size_t elementsPerThread)
{
    size_t ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(ix >= chunkLength) return;
    uint8_t *curTuple = (uint8_t *)malloc(n * sizeof(uint8_t));
    for(size_t i = 0; i<elementsPerThread; i++)
    {
        size_t curIx = elementsPerThread*ix + i;
        if(curIx >= chunkLength) return;
        for(size_t j = 0; j<n; j++)
        {
            curTuple[j] = inData[j*chunkLength + curIx];
        }
        GFMatrix aug = allocAugmentedMatrix(decodeMatrix, curTuple);
        GFGaussianElimination(&aug);
        for(size_t j = 0; j<n; j++)
        {
            outData[j*chunkLength + curIx] = aug.data[j*aug.n + n];
        }
        free(aug.data);
    }
    free(curTuple);
}