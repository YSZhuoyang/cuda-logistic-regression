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

__device__ __forceinline__ float shuffleReduceSum( float regValue )
{
    for (unsigned int shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
        regValue += __shfl_down( regValue, shift );
    // for (unsigned int i = 1; i < WARP_SIZE; i *= 2) // i =<< 1
    //     regValue += __shfl_xor( regValue, i );
    return regValue;
}

// Sum up any arrays with a maximum length of 1024
__device__ __forceinline__ float shuffleParallelSum(
    float regValue,
    const unsigned int numWarps )
{
    __shared__ float shared[32];
    int warpThreadId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // Performing warp reduction. Only the threads with 0 index
    // within the warp have the "val" value set with the warp reduction result
    regValue = shuffleReduceSum( regValue );

    // Only the threads with 0 index within the warp write the warp result to shared memory
    if (warpThreadId == 0) shared[warpId] = regValue;

    // Wait for all warp reductions
    __syncthreads();

    // There will be at most 1024 threads within a block.
    // The partial sum is read from shared memory only the corresponding
    // warp existed, otherwise the partial sum is set to zero.
    if (threadIdx.x < numWarps)
    {
        regValue = shared[warpThreadId];
        // The first warp performs the final partial warp summation.
        // Note that numWarps is always smaller than 32 given an array with a maximum length of 1024.
        if (warpId == 0) return shuffleReduceSum( regValue );
    }

    return 0;
}

// Parallel sum using a shared memory
__device__ __forceinline__ void parallelSum(
    float* __restrict__ sharedData,
    const unsigned int length )
{
    for (unsigned int i = length; i > 1; i >>= 1)
    {
        unsigned int shift = i / 2;
        if (threadIdx.x < shift)
        {
            sharedData[threadIdx.x] +=
                sharedData[threadIdx.x + shift];

            // Odd
            if (i & 1 && threadIdx.x == shift - 1)
                sharedData[threadIdx.x] += sharedData[i - 1];
        }
        __syncthreads();
    }
}

__global__ void Dot(
    float* __restrict__ dCostArr,
    const float* __restrict__ dWeightArr,
    const float* __restrict__ dFeatureMat,
    const unsigned short* __restrict__ dClassArr,
    const unsigned int numWarps,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    unsigned int instanceId = blockIdx.y * gridDim.x + blockIdx.x;
    // unsigned int featureId = threadIdx.y * blockDim.x + threadIdx.x;
    if (instanceId >= numInstances || threadIdx.x >= numFeatures) return;
    // if (threadIdx.x == 0) printf( "Instance ID: %u\n", instanceId );

    float dotProd = dWeightArr[numFeatures];
    const float* __restrict__ dFeaOffset = dFeatureMat + instanceId * numFeatures;

    dotProd += shuffleParallelSum(
        dWeightArr[threadIdx.x] * dFeaOffset[threadIdx.x],
        numWarps );

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

__global__ void UpdateWeight(
    float* __restrict__ dWeightArr,
    const float* __restrict__ dCostArr,
    const float* __restrict__ dFeatureMatTrans,
    const unsigned int alpha,
    const unsigned int chunkSize,
    const unsigned int numWarps,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    // One block per feature, one thread per group of instances
    unsigned int featureId = blockIdx.y * gridDim.x + blockIdx.x;
    // unsigned int instChunkId = threadIdx.y * blockDim.x + threadIdx.x;
    if (threadIdx.x >= numInstances || featureId >= numFeatures) return;

    unsigned int stopId;
    if (threadIdx.x == blockDim.x - 1) // Last chunk
        stopId = numInstances;
    else
        stopId = chunkSize * (threadIdx.x + 1);

    float multSum = 0.0;
    for (unsigned int i = chunkSize * threadIdx.x; i < stopId; i++)
        multSum += dFeatureMatTrans[featureId * numInstances + i] * dCostArr[i];
    multSum = shuffleParallelSum(
        multSum,
        numWarps );

    // Update weights
    if (threadIdx.x == 0)
    {
        dWeightArr[featureId] -=
            alpha / (float) numInstances * multSum;

        if (featureId == 0)
            printf( "Updating weights completed, weight: %f\n", dWeightArr[0] );
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

    float* dCostArr;
    float* dWeightArr;
    float* dFeatureMat;
    float* dFeatureMatTrans;
    unsigned short* dClassArr;
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, (numFeatures + 1) * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dCostArr, numInstances * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMat, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMatTrans, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassArr, numInstances * sizeof( unsigned short ) ) );
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

    /*----------- Determine block and grid size of Dot kernel -----------*/
    dim3 dotBlockDim;
    dim3 dotGridDim;
    // Assume numFeatures <= 1024 (max number of threads per block)
    dotBlockDim.x = numFeatures;
    if (numInstances < 1024)
        dotGridDim.x = numInstances;
    else
    {
        dotGridDim.x = 1024;
        dotGridDim.y = (numInstances + dotGridDim.x - 1) / dotGridDim.x;
    }
    // Compute number of warps for shuffle reduction sum
    unsigned int actNumWarps = (numFeatures + WARP_SIZE - 1) / WARP_SIZE;

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
    dim3 uwBlockDim;
    dim3 uwGridDim;
    unsigned int uwChunkSize;
    unsigned int uwNumChunks;
    if (numInstances > 512)
    {
        uwNumChunks = 512;
        uwChunkSize = numInstances / uwNumChunks;
    }
    else
    {
        uwNumChunks = numInstances;
        uwChunkSize = 1;
    }
    uwBlockDim.x = uwNumChunks;
    uwGridDim.x = numFeatures;
    // Compute number of warps for shuffle reduction sum
    unsigned int uwNumWarps = (uwNumChunks + WARP_SIZE - 1) / WARP_SIZE;

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
            actNumWarps,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );

        ComputeCost<<< ccGridDim, ccBlockDim >>>(
            dCostArr,
            dClassArr,
            numInstances );
        cudaErrorCheck( cudaGetLastError() );

        UpdateWeight<<< uwGridDim, uwBlockDim >>>(
            dWeightArr,
            dCostArr,
            dFeatureMatTrans,
            alpha,
            uwChunkSize,
            uwNumWarps,
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

    cudaFree( dFeatureMat );
    cudaFree( dFeatureMatTrans );
    cudaFree( dClassArr );
    cudaFree( dWeightArr );
    cudaFree( dCostArr );

    free( node.weights );

    return 0;
}
