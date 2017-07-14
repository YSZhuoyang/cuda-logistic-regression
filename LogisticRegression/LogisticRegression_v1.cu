#include "Helper.h"
#include "ArffImporter.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include "cublas_v2.h"


#define WARP_SIZE 32

Node initNode( unsigned int numFeatures )
{
    Node node;
    node.numFeatures = numFeatures;
    node.weights = (double*) malloc( (numFeatures + 1) * sizeof( double ) );
    memset( node.weights, 0, (numFeatures + 1) * sizeof( double ) );

    return node;
}

void normalize(
    std::vector<NumericAttr> featureVec,
    double* featureMat,
    double* featureMatTrans,
    unsigned int numInstances )
{
    unsigned int numFeatures = featureVec.size();

    for (unsigned int i = 0; i < numFeatures; i++)
    {
        // Use either range / standard deviation
        double range = featureVec[i].max - featureVec[i].min;
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

__device__ __forceinline__ double shuffleReduceSum( double regValue )
{
    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        regValue += __shfl_down( regValue, offset );
    // for (unsigned int i = 1; i < WARP_SIZE; i *= 2) // i =<< 1
    //     regValue += __shfl_xor( regValue, i );
    return regValue;
}

// Sum up any arrays with a maximum length of 1024
// elementId is equal to the threadId
__device__ __forceinline__ double shuffleParallelSum(
    double regValue,
    const unsigned int numWarps,
    const unsigned int elementId )
{
    __shared__ double shared[32];
    // extern __shared__ double shared[];

    int warpThreadId = elementId % WARP_SIZE;
    int warpId = elementId / WARP_SIZE;

    // Performing warp reduction. Only the threads with 0 index
    // within the warp have the "val" value set with the warp reduction result
    regValue = shuffleReduceSum( regValue );     

    // Only the threads with 0 index within the warp write the warp result to shared memory
    if (warpThreadId == 0) shared[warpId] = regValue;

    // Wait for all warp reductions
    __syncthreads();

    // There will be at most 1024 threads within a block and at most 1024 blocks within a grid.
    // The partial sum is read from shared memory only the corresponding
    // warp existed, otherwise the partial sum is set to zero.
    regValue = (elementId < numWarps) ? shared[warpThreadId] : 0;

    // The first warp performs the final partial warp summation.
    // Note that numWarps is always smaller than 32 given an array with a maximum length of 1024.
    if (warpId == 0) regValue = shuffleReduceSum( regValue ); 

    return regValue;
}

// Parallel sum using a shared memory
__device__ __forceinline__ void parallelSum(
    double* __restrict__ sharedData,
    const unsigned int elementId,
    const unsigned int length )
{
    for (unsigned int i = length; i > 1; i >>= 1)
    {
        unsigned int shift = i / 2;
        if (elementId < shift)
        {
            sharedData[elementId] +=
                sharedData[elementId + shift];

            // Odd
            if (i & 1 && elementId == shift - 1)
                sharedData[elementId] += sharedData[i - 1];
        }
        __syncthreads();
    }
}

__global__ void Activate(
    double* __restrict__ dDiffArr,
    const double* __restrict__ dWeightArr,
    const double* __restrict__ dfeatureMat,
    const unsigned short* __restrict__ dClassArr,
    const unsigned int numWarps,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    unsigned int instanceId = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int featureId = threadIdx.y * blockDim.x + threadIdx.x;
    if (instanceId >= numInstances || featureId >= numFeatures) return;
    // if (featureId == 0) printf( "Instance ID: %u\n", instanceId );

    double hRes = dWeightArr[numFeatures];
    const double* dFeaOffset = dfeatureMat + instanceId * numFeatures;
    // extern __shared__ double dProductShared[];
    // dProductShared[featureId] =
    //     dWeightArr[featureId] * dFeaOffset[featureId];
    // __syncthreads();

    // Assume numFeatures is big
    // parallelSum( dProductShared, featureId, numFeatures );
    hRes += shuffleParallelSum(
        dWeightArr[featureId] * dFeaOffset[featureId],
        numWarps,
        featureId );

    if (featureId == 0)
    {
        // hRes += dProductShared[0];
        hRes = 1.0 / (1.0 + exp(-hRes));
        dDiffArr[instanceId] = hRes - (double) dClassArr[instanceId];
    }
}

__global__ void UpdateWeight(
    double* __restrict__ dWeightArr,
    const double* __restrict__ dDiffArr,
    const double* __restrict__ dfeatureMatTrans,
    const unsigned int alpha,
    const unsigned int chunkSize,
    const unsigned int numWarps,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    // One block per feature, one thread per group of instances
    unsigned int featureId = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int instChunkId = threadIdx.y * blockDim.x + threadIdx.x;
    if (instChunkId >= numInstances || featureId >= numFeatures) return;

    unsigned int stopId;
    if (instChunkId == blockDim.x - 1) // Last chunk
        stopId = numInstances;
    else
        stopId = chunkSize * (instChunkId + 1);

    double multSum = 0.0;
    // Values of one feature
    // extern __shared__ double dProductShared[];
    // for (unsigned int i = chunkSize * instChunkId; i < stopId; i++)
    //     multSum += dfeatureMatTrans[featureId * numInstances + i] * dDiffArr[i];
    // dProductShared[instChunkId] = multSum;
    // __syncthreads();

    for (unsigned int i = chunkSize * instChunkId; i < stopId; i++)
        multSum += dfeatureMatTrans[featureId * numInstances + i] * dDiffArr[i];
    multSum = shuffleParallelSum(
        multSum,
        numWarps,
        instChunkId );

    // Assume numInstances is big
    // parallelSum( dProductShared, instChunkId, blockDim.x );

    // Update weights
    if (instChunkId == 0)
    {
        dWeightArr[featureId] -=
            // alpha / (double) numInstances * dProductShared[0];
            alpha / (double) numInstances * multSum;

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

// void cublasErrorCheck( cublasStatus_t cublasRes )
// {
//     if (cublasRes != CUBLAS_STATUS_SUCCESS)
//         printf( "Cublas library failed to load.\n" );
// }

int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    unsigned int numInstances = trainSetImporter.GetNumInstances();
    double* featureMat = trainSetImporter.GetFeatureMat();
    double* featureMatTrans = trainSetImporter.GetFeatureMatTrans();
    unsigned short* classArr = trainSetImporter.GetClassIndex();
    std::vector<NumericAttr> featureVec = trainSetImporter.GetFeatures();
    unsigned int numFeatures = featureVec.size();

    unsigned int alpha = 50;
    unsigned int maxIter = 200;
    unsigned int iter = 0;

    normalize( featureVec, featureMat, featureMatTrans, numInstances );
    Node node = initNode( numFeatures );

    double* dDiffArr;
    double* dWeightArr;
    double* dfeatureMat;
    double* dfeatureMatTrans;
    unsigned short* dClassArr;
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, (numFeatures + 1) * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dDiffArr, numInstances * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dfeatureMat, numInstances * numFeatures * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dfeatureMatTrans, numInstances * numFeatures * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassArr, numInstances * sizeof( unsigned short ) ) );

    cudaErrorCheck( cudaMemcpy(
        dfeatureMat,
        featureMat,
        numInstances * numFeatures * sizeof( double ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy(
        dfeatureMatTrans,
        featureMatTrans,
        numInstances * numFeatures * sizeof( double ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy(
        dWeightArr,
        node.weights,
        (numFeatures + 1) * sizeof( double ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy(
        dClassArr,
        classArr,
        numInstances * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    // cublasHandle_t cublasHandle;
    // cublasErrorCheck( cublasCreate( &cublasHandle ) );

    // Determine block and grid size of Activat kernel
    dim3 actBlockDim;
    dim3 actGridDim;
    dim3 uwBlockDim;
    dim3 uwGridDim;
    // Assume numFeatures <= 1024 (max number of threads per block)
    actBlockDim.x = numFeatures;
    if (numInstances < 1024)
        actGridDim.x = numInstances;
    else
    {
        actGridDim.x = 1024;
        actGridDim.y = (numInstances + actGridDim.x - 1) / actGridDim.x;
    }

    // Determine block and grid size of UpdateWeight kernel
    uwBlockDim.x = actGridDim.x;
    uwGridDim.x = actBlockDim.x;
    unsigned int uwChunkSize = numInstances / uwBlockDim.x;

    // Compute number of warps for shuffle reduction sum
    unsigned int numWarps = (actBlockDim.x % WARP_SIZE > 0) ?
        actBlockDim.x / WARP_SIZE + 1 : actBlockDim.x / WARP_SIZE;

    time_t start, end;
    double dif;
    time( &start );
    
    printf( "\nStart gradient descent...\n" );

    // Gradient descent
    do
    {
        Activate<<< actGridDim, actBlockDim >>>(
            dDiffArr,
            dWeightArr,
            dfeatureMat,
            dClassArr,
            numWarps,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );

        UpdateWeight<<< uwGridDim, uwBlockDim >>>(
            dWeightArr,
            dDiffArr,
            dfeatureMatTrans,
            alpha,
            uwChunkSize,
            numWarps,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );

        iter++;
    }
    while (iter == 1 || iter < maxIter);

    cudaErrorCheck( cudaThreadSynchronize() );
    
    // cudaMemcpy(weight);
    // cublasErrorCheck( cublasDestroy( cublasHandle ) );

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken is %.2lf seconds.\n", dif );

    cudaFree( dfeatureMat );
    cudaFree( dfeatureMatTrans );
    cudaFree( dClassArr );
    cudaFree( dWeightArr );
    cudaFree( dDiffArr );

    free( node.weights );

    return 0;
}
