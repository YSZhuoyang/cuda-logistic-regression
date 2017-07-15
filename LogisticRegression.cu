#include "Helper.h"
#include "ArffImporter.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#define WARP_SIZE 32

Node initNode( unsigned int numFeatures )
{
    Node node;
    node.numFeatures = numFeatures;
    node.weights = (float*) malloc( (numFeatures + 1) * sizeof( float ) );
    memset( node.weights, 0, (numFeatures + 1) * sizeof( float ) );

    return node;
}

void normalize(
    std::vector<NumericAttr> featureVec,
    float* featureMat,
    float* featureMatTrans,
    unsigned int numInstances )
{
    unsigned int numFeatures = featureVec.size();

    for (unsigned int i = 0; i < numFeatures; i++)
    {
        // Use either range / standard deviation
        float range = featureVec[i].max - featureVec[i].min;
        if (range == 0.0) continue;

        for (unsigned int j = 0; j < numInstances; j++)
        {
            unsigned int featureIndex = j * numFeatures + i;
            featureMat[featureIndex] =
                (featureMat[featureIndex] - featureVec[i].mean) / range;
            featureMatTrans[i * numInstances + j] = featureMat[featureIndex];
        }
    }
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum512(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

    if (threadIdx.x < 256)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 256];
    __syncthreads();

    if (threadIdx.x < 128)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 128];
    __syncthreads();

    if (threadIdx.x < 64)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        // Compile unroll for loop?
        for (unsigned short shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
            sum += __shfl_down( sum, shift );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum256(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

    if (threadIdx.x < 128)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 128];
    __syncthreads();

    if (threadIdx.x < 64)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        // Compile unroll for loop?
        for (unsigned short shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
            sum += __shfl_down( sum, shift );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum128(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

    if (threadIdx.x < 64)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        // Compile unroll for loop?
        for (unsigned short shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
            sum += __shfl_down( sum, shift );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum64(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        // Compile unroll for loop?
        for (unsigned short shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
            sum += __shfl_down( sum, shift );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

__device__ __forceinline__ float parallelSum32( float sum )
{
    // Reduce final warp using shuffle
    // Compile unroll for loop?
    for (unsigned short shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
        sum += __shfl_down( sum, shift );

    return sum;
}

__global__ void Dot(
    float* __restrict__ dCostArr,
    const float* __restrict__ dWeightArr,
    const float* __restrict__ dFeatureMat,
    const unsigned short* __restrict__ dClassArr,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    unsigned int instanceId = blockIdx.y * gridDim.x + blockIdx.x;
    if (instanceId >= numInstances) return;
    // if (threadIdx.x == 0) printf( "Instance ID: %u\n", instanceId );

    float dotProd = dWeightArr[numFeatures];
    const float* __restrict__ dFeaOffset = dFeatureMat + instanceId * numFeatures;

    __shared__ float sharedProd[512];
    unsigned int offset = threadIdx.x * 2;
    float partialSum = 0.0f;
    if (offset < numFeatures)
        partialSum += dWeightArr[offset] * dFeaOffset[offset];
    // else return;
    if (offset + 1 < numFeatures)
        partialSum += dWeightArr[offset + 1] * dFeaOffset[offset + 1];
    sharedProd[threadIdx.x] = partialSum;
    __syncthreads();

    dotProd += parallelSum512( sharedProd );
    if (threadIdx.x == 0) dCostArr[instanceId] = dotProd;
}

__global__ void ComputeCost(
    float* __restrict__ dCostArr,
    const unsigned short* __restrict__ dClassArr,
    const unsigned int numInstances )
{
    unsigned int instanceId = blockIdx.x * blockDim.x + threadIdx.x;
    if (instanceId >= numInstances) return;

    float cost = dCostArr[instanceId];
    cost = 1.0 / (1.0 + exp(-cost)) - (float) dClassArr[instanceId];
    dCostArr[instanceId] = cost;
}

__global__ void UpdateWeightL2(
    float* __restrict__ dWeightArr,
    const float* __restrict__ dPartSumArr,
    const unsigned int alpha,
    const unsigned int partSumLen,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    if (blockIdx.x >= numFeatures) return;

    // extern __shared__ float sharedPartSum[];
    // unsigned int offset = threadIdx.x * 2;
    // float sum = 0.0f;
    // if (offset < partSumLen)
    //     sum += dPartSumArr[offset];
    // // else return;
    // if (offset + 1 < partSumLen)
    //     sum += dPartSumArr[offset + 1];
    // sharedPartSum[threadIdx.x] = sum;
    // __syncthreads();

    // sum = parallelSum( sharedPartSum );

    float sum = (threadIdx.x >= partSumLen) ?
        0.0f : dPartSumArr[blockIdx.x * partSumLen + threadIdx.x];
    sum = parallelSum32( sum );
    // Update weights
    if (threadIdx.x == 0)
    {
        dWeightArr[blockIdx.x] -=
            alpha / (float) numInstances * sum;

        if (blockIdx.x == 0)
            printf( "Updating weights completed, weight: %f\n", dWeightArr[0] );
    }
}

__global__ void UpdateWeight(
    float* __restrict__ dWeightArr,
    float* __restrict__ dPartSumArr,
    const float* __restrict__ dCostArr,
    const float* __restrict__ dFeatureMatTrans,
    const unsigned int alpha,
    const unsigned int partSumLen,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    unsigned int featureId;
    unsigned int offset;
    if (partSumLen == 1)
    {
        featureId = blockIdx.x;
        offset = threadIdx.x * 2;
    }
    else
    {
        featureId = blockIdx.y;
        offset = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    }

    // unsigned int featureId = blockIdx.y;
    // unsigned int offset = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (featureId >= numFeatures) return;

    const float* __restrict__ dFeaMatTransOffset =
        dFeatureMatTrans + featureId * numInstances;
    __shared__ float sharedProd[512];
    float partialSum = 0.0;
    if (offset < numInstances)
        partialSum += dFeaMatTransOffset[offset] * dCostArr[offset];
    if (offset + 1 < numInstances)
        partialSum += dFeaMatTransOffset[offset + 1] * dCostArr[offset + 1];
    sharedProd[threadIdx.x] = partialSum;
    __syncthreads();

    partialSum = parallelSum512( sharedProd );
    if (threadIdx.x == 0)
        dPartSumArr[featureId * partSumLen + blockIdx.x] = partialSum;

    // One block per feature, one thread per group of instances
    // unsigned int featureId = blockIdx.y * gridDim.x + blockIdx.x;
    // // unsigned int instChunkId = threadIdx.y * blockDim.x + threadIdx.x;
    // if (threadIdx.x >= numInstances || featureId >= numFeatures) return;

    // unsigned int stopId;
    // if (threadIdx.x == blockDim.x - 1) // Last chunk
    //     stopId = numInstances;
    // else
    //     stopId = chunkSize * (threadIdx.x + 1);

    // float dotProd = 0.0;
    // for (unsigned int i = chunkSize * threadIdx.x; i < stopId; i++)
    //     dotProd += dFeatureMatTrans[featureId * numInstances + i] * dCostArr[i];

    // __shared__ float sharedProd[512];
    // sharedProd[threadIdx.x] = dotProd;
    // __syncthreads();

    // dotProd = parallelSum( sharedProd );

    // // Update weights
    if (partSumLen == 1 && threadIdx.x == 0)
    {
        dWeightArr[featureId] -=
            alpha / (float) numInstances * partialSum;

        if (featureId == 0)
            printf( "Updating weights completed, weight: %f\n", partialSum );
    }
}

inline void cudaErrorCheck( cudaError_t cudaRes )
{
    if (cudaRes != cudaSuccess)
        printf(
            "kernel launch failed with error \"%s\".\n",
            cudaGetErrorString( cudaRes )
        );
}

int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    unsigned int numInstances = trainSetImporter.GetNumInstances();
    float* featureMat = trainSetImporter.GetFeatureMat();
    float* featureMatTrans = trainSetImporter.GetFeatureMatTrans();
    unsigned short* classArr = trainSetImporter.GetClassIndex();
    std::vector<NumericAttr> featureVec = trainSetImporter.GetFeatures();
    unsigned int numFeatures = featureVec.size();

    normalize( featureVec, featureMat, featureMatTrans, numInstances );
    Node node = initNode( numFeatures );

    /*----------- Determine block and grid size of Dot kernel -----------*/
    dim3 dotBlockDim;
    dim3 dotGridDim;
    // Assume numFeatures <= 1024 (max number of threads per block)
    // dotBlockDim.x = numFeatures;
    dotBlockDim.x = 512;
    if (numInstances < 1024)
        dotGridDim.x = numInstances;
    else
    {
        dotGridDim.x = 1024;
        dotGridDim.y = (numInstances + dotGridDim.x - 1) / dotGridDim.x;
    }

    /*------- Determine block and grid size of ComputeCost kernel -------*/
    dim3 ccBlockDim;
    dim3 ccGridDim;
    if (numInstances > 1024)
    {
        ccBlockDim.x = 1024;
        ccGridDim.x = (numInstances + 1023) / 1024;
    }
    else ccBlockDim.x = numInstances;

    /*------- Determine block and grid size of UpdateWeight kernel -------*/
    dim3 uwBlockDimL1;
    dim3 uwGridDimL1;
    dim3 uwBlockDimL2;
    dim3 uwGridDimL2;
    unsigned int partSumLen;
    // Assume numFeatures < 1024
    if (numInstances > 1024)
    {
        uwBlockDimL1.x = 512;
        uwGridDimL1.x = (numInstances + 1023) / 1024;
        uwGridDimL1.y = numFeatures;
        partSumLen = uwGridDimL1.x;

        // Assume partSumLen <= 1024
        if (partSumLen < 64) uwBlockDimL2.x = 32;
        else if (partSumLen < 128) uwBlockDimL2.x = 64;
        else if (partSumLen < 256) uwBlockDimL2.x = 128;
        else if (partSumLen < 512) uwBlockDimL2.x = 256;
        else uwBlockDimL2.x = 512;
        uwGridDimL2.x = numFeatures;
    }
    else
    {
        partSumLen = 1;
        if (numInstances < 64) uwBlockDimL1.x = 32;
        else if (numInstances < 128) uwBlockDimL1.x = 64;
        else if (numInstances < 256) uwBlockDimL1.x = 128;
        else if (numInstances < 512) uwBlockDimL1.x = 256;
        uwGridDimL1.x = numFeatures;
    }

    float* dCostArr;
    float* dWeightArr;
    float* dFeatureMat;
    float* dFeatureMatTrans;
    float* dPartSumArr;
    unsigned short* dClassArr;
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, (numFeatures + 1) * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dCostArr, numInstances * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMat, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMatTrans, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassArr, numInstances * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dPartSumArr, partSumLen * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dFeatureMat,
        featureMat,
        numInstances * numFeatures * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dFeatureMatTrans,
        featureMatTrans,
        numInstances * numFeatures * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dWeightArr,
        node.weights,
        (numFeatures + 1) * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassArr,
        classArr,
        numInstances * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    // Gradient descent params
    unsigned int alpha = 50;
    unsigned int maxIter = 200;
    unsigned int iter = 0;

    time_t start, end;
    float dif;
    time( &start );
    
    printf( "\nStart gradient descent...\n" );

    // Gradient descent
    while (iter++ < maxIter)
    {
        Dot<<< dotGridDim, dotBlockDim >>>(
            dCostArr,
            dWeightArr,
            dFeatureMat,
            dClassArr,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );

        ComputeCost<<< ccGridDim, ccBlockDim >>>(
            dCostArr,
            dClassArr,
            numInstances );
        cudaErrorCheck( cudaGetLastError() );

        UpdateWeight<<< uwGridDimL1, uwBlockDimL1 >>>(
            dWeightArr,
            dPartSumArr,
            dCostArr,
            dFeatureMatTrans,
            alpha,
            partSumLen,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );
        if (partSumLen > 1)
        {
            UpdateWeightL2<<< uwGridDimL2, uwBlockDimL2 >>>(
                dWeightArr,
                dPartSumArr,
                alpha,
                partSumLen,
                numInstances,
                numFeatures );
            cudaErrorCheck( cudaGetLastError() );
        }
    }

    cudaErrorCheck( cudaThreadSynchronize() );
    
    // cudaMemcpy(weight);
    // cublasErrorCheck( cublasDestroy( cublasHandle ) );

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken is %.2lf seconds.\n", dif );

    cudaFree( dFeatureMat );
    cudaFree( dFeatureMatTrans );
    cudaFree( dClassArr );
    cudaFree( dWeightArr );
    cudaFree( dCostArr );
    cudaFree( dPartSumArr );
    free( node.weights );

    return 0;
}
