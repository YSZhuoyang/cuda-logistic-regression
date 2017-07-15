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

__device__ __forceinline__ float shuffleSum32( float sum )
{
    // Reduce final warp using shuffle
    for (unsigned short shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
        sum += __shfl_down( sum, shift );

    return sum;
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
        sum = shuffleSum32( sum );
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
        sum = shuffleSum32( sum );
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
        sum = shuffleSum32( sum );
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
        sum = shuffleSum32( sum );
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

__global__ void VecDotMat(
    float* __restrict__ dResVec,
    const float* __restrict__ dVec,
    const float* __restrict__ dMat,
    const unsigned int numColumns,
    const unsigned int numRows )
{
    unsigned int rowId = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int columnId = threadIdx.x * 2;
    if (rowId >= numRows) return;

    const float* __restrict__ dRow = dMat + rowId * numColumns;
    float sum = 0.0f;
    if (numColumns < 32 && blockDim.x == 32)
    {
        if (threadIdx.x < numColumns)
            sum = dRow[threadIdx.x] * dVec[threadIdx.x];
        sum = shuffleSum32( sum );
    }
    else
    {
        extern __shared__ float sharedPartSum[];
        if (columnId < numColumns)
            sum += dRow[columnId] * dVec[columnId];
        if (columnId + 1 < numColumns)
            sum += dRow[columnId + 1] * dVec[columnId + 1];
        sharedPartSum[threadIdx.x] = sum;
        __syncthreads();

        switch (blockDim.x)
        {
            case 64:
                sum = parallelSum64( sharedPartSum );
                break;
            case 128:
                sum = parallelSum128( sharedPartSum );
                break;
            case 256:
                sum = parallelSum256( sharedPartSum );
                break;
            case 512:
                sum = parallelSum512( sharedPartSum );
                break;
            default:
                break;
        }
    }

    if (threadIdx.x == 0) dResVec[rowId] = sum;
}

__global__ void VecDotMatPartial(
    float* __restrict__ dPartSumArr,
    const float* __restrict__ dVec,
    const float* __restrict__ dMat,
    const unsigned int partSumLen,
    const unsigned int numColumns,
    const unsigned int numRows )
{
    unsigned int rowId = blockIdx.y;
    unsigned int columnId = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (rowId >= numRows) return;

    const float* __restrict__ dRow = dMat + rowId * numColumns;
    float sum = 0.0;

    extern __shared__ float sharedPartSum[];
    if (columnId < numColumns)
        sum += dRow[columnId] * dVec[columnId];
    if (columnId + 1 < numColumns)
        sum += dRow[columnId + 1] * dVec[columnId + 1];
    sharedPartSum[threadIdx.x] = sum;
    __syncthreads();

    sum = parallelSum512( sharedPartSum );
    if (threadIdx.x == 0)
        dPartSumArr[partSumLen * rowId + blockIdx.x] = sum;
}

// Reduce partial sum
__global__ void SumColumns(
    float* __restrict__ dResVec,
    const float* __restrict__ dMat,
    const unsigned int numColumns,
    const unsigned int numRows )
{
    if (blockIdx.x >= numRows) return;

    float sum = 0.0f;
    if (blockDim.x == 32)
    {
        if (threadIdx.x < numColumns)
            sum = dMat[blockIdx.x * numColumns + threadIdx.x];
        sum = shuffleSum32( sum );
    }
    else
    {
        extern __shared__ float sharedPartSum[];
        unsigned int columnId = threadIdx.x * 2;
        if (columnId < numColumns)
            sum += dMat[columnId];
        if (columnId + 1 < numColumns)
            sum += dMat[columnId + 1];
        sharedPartSum[threadIdx.x] = sum;
        __syncthreads();

        switch (blockDim.x)
        {
            case 64:
                sum = parallelSum64( sharedPartSum );
                break;
            case 128:
                sum = parallelSum128( sharedPartSum );
                break;
            case 256:
                sum = parallelSum256( sharedPartSum );
                break;
            case 512:
                sum = parallelSum512( sharedPartSum );
                break;
            default:
                break;
        }
    }

    if (threadIdx.x == 0) dResVec[blockIdx.x] = sum;
}

__global__ void UpdateWeight(
    float* __restrict__ dWeightArr,
    const float* __restrict__ dSumArr,
    const unsigned int alpha,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    unsigned int featureId = blockIdx.x * blockDim.x + threadIdx.x;
    if (featureId >= numFeatures) return;

    dWeightArr[featureId] -=
        alpha / (float) numInstances * dSumArr[featureId];

    if (featureId == 0)
        printf( "Updating weights completed, weight: %f\n", dWeightArr[0] );
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

    /*-- Determine block and grid size of Weight Dot FeatureMat kernel --*/
    // dim3 preBlockDimL1;
    // dim3 preGridDimL1;
    // dim3 preBlockDimL2;
    // dim3 preGridDimL2;
    // unsigned int prePartSumLen;
    // unsigned int preSharedMemoSizeL1;
    // unsigned int preSharedMemoSizeL2;
    // // Assume numFeatures < 1024
    // if (numFeatures > 1024)
    // {
    //     preBlockDimL1.x = 512;
    //     uwGridDimL1.x = (numFeatures + 1023) / 1024;
    //     uwGridDimL1.y = numInstances;
    //     prePartSumLen = uwGridDimL1.x;
    //     sharedMemoSizeL1 = 512 * sizeof( float );

    //     uwGridDimL2.x = numInstances;
    //     // Assume uwPartSumLen <= 1024
    //     if (uwPartSumLen <= 32)
    //     {
    //         uwBlockDimL2.x = 32;
    //         sharedMemoSizeL2 = 0;
    //     }
    //     else
    //     {
    //         if (prePartSumLen <= 64) uwBlockDimL2.x = 32;
    //         else if (prePartSumLen <= 128) uwBlockDimL2.x = 64;
    //         else if (prePartSumLen <= 256) uwBlockDimL2.x = 128;
    //         else if (prePartSumLen <= 512) uwBlockDimL2.x = 256;
    //         else uwBlockDimL2.x = 512;
    //         sharedMemoSizeL2 = uwBlockDimL2.x * sizeof( float );
    //     }
    // }
    // else
    // {
    //     uwPartSumLen = 1;
    //     uwGridDimL1.x = numFeatures;
    //     if (numFeatures <= 32)
    //     {
    //         preBlockDimL1.x = 32;
    //         sharedMemoSizeL1 = 0;
    //     }
    //     else
    //     {
    //         if (numFeatures <= 64) preBlockDimL1.x = 32;
    //         else if (numFeatures <= 128) preBlockDimL1.x = 64;
    //         else if (numFeatures <= 256) preBlockDimL1.x = 128;
    //         else if (numFeatures <= 512) preBlockDimL1.x = 256;
    //         sharedMemoSizeL1 = preBlockDimL1.x * sizeof( float );
    //     }
    // }

    dim3 dotBlockDim;
    dim3 dotGridDim;
    unsigned int preSharedMemoSizeL1 = 512 * sizeof( float );
    unsigned int prePartSumLen = 1;
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

    /* Determine block and grid size of ComputeCost & UpdateWeight kernel */
    dim3 uwBlockDim;
    dim3 uwGridDim;
    if (numFeatures > 1024)
    {
        uwBlockDim.x = 1024;
        uwGridDim.x = (numInstances + 1023) / 1024;
    }
    else uwBlockDim.x = numFeatures;

    /* Determine block and grid size of ComputeCost & UpdateWeight kernel */
    dim3 ccBlockDim;
    dim3 ccGridDim;
    if (numInstances > 1024)
    {
        ccBlockDim.x = 1024;
        ccGridDim.x = (numInstances + 1023) / 1024;
    }
    else ccBlockDim.x = numInstances;

    /*- Determine block and grid size of Cost Dot FeatureMatTrans kernel -*/
    dim3 uwBlockDimL1;
    dim3 uwGridDimL1;
    dim3 uwBlockDimL2;
    dim3 uwGridDimL2;
    unsigned int uwPartSumLen;
    unsigned int uwSharedMemoSizeL1;
    unsigned int uwSharedMemoSizeL2;
    // Assume numFeatures < 1024
    if (numInstances > 1024)
    {
        uwBlockDimL1.x = 512;
        uwGridDimL1.x = (numInstances + 1023) / 1024;
        uwGridDimL1.y = numFeatures;
        uwPartSumLen = uwGridDimL1.x;
        uwSharedMemoSizeL1 = 512 * sizeof( float );

        uwGridDimL2.x = numFeatures;
        // Assume uwPartSumLen <= 1024
        if (uwPartSumLen <= 32)
        {
            uwBlockDimL2.x = 32;
            uwSharedMemoSizeL2 = 0;
        }
        else
        {
            if (uwPartSumLen <= 64) uwBlockDimL2.x = 32;
            else if (uwPartSumLen <= 128) uwBlockDimL2.x = 64;
            else if (uwPartSumLen <= 256) uwBlockDimL2.x = 128;
            else if (uwPartSumLen <= 512) uwBlockDimL2.x = 256;
            else uwBlockDimL2.x = 512;
            uwSharedMemoSizeL2 = uwBlockDimL2.x * sizeof( float );
        }
    }
    else
    {
        uwPartSumLen = 1;
        uwGridDimL1.x = numFeatures;
        if (numInstances <= 32)
        {
            uwBlockDimL1.x = 32;
            uwSharedMemoSizeL1 = 0;
        }
        else
        {
            if (numInstances <= 64) uwBlockDimL1.x = 32;
            else if (numInstances <= 128) uwBlockDimL1.x = 64;
            else if (numInstances <= 256) uwBlockDimL1.x = 128;
            else if (numInstances <= 512) uwBlockDimL1.x = 256;
            uwSharedMemoSizeL1 = uwBlockDimL1.x * sizeof( float );
        }
    }

    float* dCostArr;
    float* dWeightArr;
    float* dFeatureMat;
    float* dFeatureMatTrans;
    float* dPrePartSumArr;
    float* dUWPartSumArr;
    float* dUWSumArr;
    unsigned short* dClassArr;
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, (numFeatures + 1) * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dCostArr, numInstances * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMat, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMatTrans, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassArr, numInstances * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dUWSumArr, numFeatures * sizeof( float ) ) );
    if (uwPartSumLen > 1)
        cudaErrorCheck( cudaMalloc( (void**) &dUWPartSumArr, uwPartSumLen * numFeatures * sizeof( float ) ) );
    if (prePartSumLen > 1)
        cudaErrorCheck( cudaMalloc( (void**) &dPrePartSumArr, prePartSumLen * numInstances * sizeof( float ) ) );
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
        // Predict
        VecDotMat<<< dotGridDim, dotBlockDim, preSharedMemoSizeL1 >>>(
            dCostArr,
            dWeightArr,
            dFeatureMat,
            numFeatures,
            numInstances );
        cudaErrorCheck( cudaGetLastError() );

        ComputeCost<<< ccGridDim, ccBlockDim >>>(
            dCostArr,
            dClassArr,
            numInstances );
        cudaErrorCheck( cudaGetLastError() );

        // Update weights
        if (uwPartSumLen == 1)
        {
            VecDotMat<<< uwGridDimL1, uwBlockDimL1, uwSharedMemoSizeL1 >>>(
                dUWSumArr,
                dCostArr,
                dFeatureMatTrans,
                numInstances,
                numFeatures );
            cudaErrorCheck( cudaGetLastError() );
        }
        else
        {
            VecDotMatPartial<<< uwGridDimL1, uwBlockDimL1, uwSharedMemoSizeL1 >>>(
                dUWPartSumArr,
                dCostArr,
                dFeatureMatTrans,
                uwPartSumLen,
                numInstances,
                numFeatures );
            cudaErrorCheck( cudaGetLastError() );
            SumColumns<<< uwGridDimL2, uwBlockDimL2, uwSharedMemoSizeL2 >>>(
                dUWSumArr,
                dUWPartSumArr,
                uwPartSumLen,
                numFeatures );
            cudaErrorCheck( cudaGetLastError() );
        }
        UpdateWeight<<< uwGridDim, uwBlockDim >>>(
            dWeightArr,
            dUWSumArr,
            alpha,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );
    }

    cudaErrorCheck( cudaThreadSynchronize() );
    
    // cudaMemcpy(weight);
    // cublasErrorCheck( cublasDestroy( cublasHandle ) );

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken is %.2lf seconds.\n", dif );

    if (uwPartSumLen > 1) cudaFree( dUWPartSumArr );
    if (prePartSumLen > 1) cudaFree( dPrePartSumArr );
    cudaFree( dFeatureMat );
    cudaFree( dFeatureMatTrans );
    cudaFree( dClassArr );
    cudaFree( dWeightArr );
    cudaFree( dCostArr );
    cudaFree( dUWSumArr );
    free( node.weights );

    return 0;
}
