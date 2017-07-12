#include "Helper.h"
#include "ArffImporter.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include "cublas_v2.h"


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
    double* featureBuff,
    double* featureBuffTrans,
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
            featureBuff[featureIndex] =
                (featureBuff[featureIndex] - featureVec[i].mean) / range;
            featureBuffTrans[i * numInstances + j] = featureBuff[featureIndex];
        }
    }
}

__device__ __forceinline__ void parallelSum(
    double* sharedData,
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
    double* dDiffArr,
    double* dWeightArr,
    const double* dFeatureBuff,
    const unsigned short* dClassBuff,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    unsigned int instanceId = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int featureId = threadIdx.y * blockDim.x + threadIdx.x;
    if (instanceId >= numInstances || featureId >= numFeatures) return;
    // if (featureId == 0) printf( "Instance ID: %u\n", instanceId );

    double hRes = dWeightArr[numFeatures];
    const double* dFeaOffset = dFeatureBuff + instanceId * numFeatures;
    extern __shared__ double dProductShared[];
    dProductShared[featureId] =
        dWeightArr[featureId] * dFeaOffset[featureId];
    __syncthreads();

    // Assume numFeatures is big
    parallelSum( dProductShared, featureId, numFeatures );

    if (featureId == 0)
    {
        hRes += dProductShared[0];
        hRes = 1.0 / (1.0 + exp(-hRes));
        dDiffArr[instanceId] = hRes - (double) dClassBuff[instanceId];
    }
}

__global__ void UpdateWeight(
    double* dDiffArr,
    double* dWeightArr,
    const double* dFeatureBuffTrans,
    const unsigned int alpha,
    const unsigned int chunkSize,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    // One block per feature, one thread per group of instances
    unsigned int featureId = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int instChunkId = threadIdx.y * blockDim.x + threadIdx.x;
    if (instChunkId >= numInstances || featureId >= numFeatures) return;

    unsigned int stopId;
    // Last chunk
    if (instChunkId == blockDim.x - 1)
        stopId = numInstances;
    else
        stopId = chunkSize * (instChunkId + 1);

    double multSum = 0.0;
    // Values of one feature
    extern __shared__ double dProductShared[];
    for (unsigned int i = chunkSize * instChunkId; i < stopId; i++)
        multSum += dFeatureBuffTrans[featureId * numInstances + i] * dDiffArr[i];
    dProductShared[instChunkId] = multSum;
    __syncthreads();

    // Assume numInstances is big
    parallelSum( dProductShared, instChunkId, blockDim.x );

    // Update weights
    if (instChunkId == 0)
    {
        dWeightArr[featureId] -=
            alpha / (double) numInstances * dProductShared[0];

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
    double* featureBuff = trainSetImporter.GetFeatureBuff();
    double* featureBuffTrans = trainSetImporter.GetFeatureBuffTrans();
    unsigned short* classIndexBuff = trainSetImporter.GetClassIndex();
    std::vector<NumericAttr> featureVec = trainSetImporter.GetFeatures();
    unsigned int numFeatures = featureVec.size();

    unsigned int alpha = 50;
    unsigned int maxIter = 200;
    unsigned int iter = 0;

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
    uwGridDim = actBlockDim;
    unsigned int uwChunkSize = numInstances / uwBlockDim.x;

    normalize( featureVec, featureBuff, featureBuffTrans, numInstances );
    Node node = initNode( numFeatures );

    double* dDiffArr;
    double* dWeightArr;
    double* dFeatureBuff;
    double* dFeatureBuffTrans;
    unsigned short* dClassBuff;
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, (numFeatures + 1) * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dDiffArr, numInstances * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureBuff, numInstances * numFeatures * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureBuffTrans, numInstances * numFeatures * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassBuff, numInstances * sizeof( unsigned short ) ) );

    cudaErrorCheck( cudaMemcpy(
        dFeatureBuff,
        featureBuff,
        numInstances * numFeatures * sizeof( double ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy(
        dFeatureBuffTrans,
        featureBuffTrans,
        numInstances * numFeatures * sizeof( double ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy(
        dWeightArr,
        node.weights,
        (numFeatures + 1) * sizeof( double ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpy(
        dClassBuff,
        classIndexBuff,
        numInstances * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    // cublasHandle_t cublasHandle;
    // cublasErrorCheck( cublasCreate( &cublasHandle ) );

    time_t start, end;
    double dif;
    time( &start );
    
    printf( "\nStart gradient descent...\n" );

    // Gradient descent
    do
    {
        Activate<<< actGridDim, actBlockDim, numFeatures * sizeof( double ) >>>(
            dDiffArr,
            dWeightArr,
            dFeatureBuff,
            dClassBuff,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );

        UpdateWeight<<< uwGridDim, uwBlockDim, uwBlockDim.x * sizeof( double ) >>>(
            dDiffArr,
            dWeightArr,
            dFeatureBuffTrans,
            alpha,
            uwChunkSize,
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

    cudaFree( dFeatureBuff );
    cudaFree( dFeatureBuffTrans );
    cudaFree( dClassBuff );
    cudaFree( dWeightArr );
    cudaFree( dDiffArr );

    free( node.weights );

    return 0;
}
