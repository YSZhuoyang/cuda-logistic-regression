
#include "Helper.h"
#include "ArffImporter.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void normalize(
    vector<NumericAttr> featureVec,
    double* featureBuff,
    unsigned int numInstance )
{
    unsigned int numFeatures = featureVec.size();

    for (unsigned int i = 0; i < numFeatures; i++)
    {
        // Use either range / standard deviation
        double range = featureVec[i].max - featureVec[i].min;
        if (range == 0.0) continue;

        for (unsigned int j = 0; j < numInstance; j++)
        {
            unsigned int featureIndex = j * numFeatures + i;
            featureBuff[featureIndex] =
                (featureBuff[featureIndex] - featureVec[i].mean) / range;
        }
    }
}

__device__ double activate(
    Node* node,
    double* inputArr )
{
    double linearRes = node->weights[node->numFeatures];
    node->inputs = inputArr;

    unsigned int numFeatures = node->numFeatures;
    for (unsigned int i = 0; i < numFeatures; i++)
        linearRes += node->weights[i] * node->inputs[i];

    node->output = 1.0 / (1.0 + exp(-linearRes));

    return node->output;
}

__device__ double computeCost( double hRes, unsigned short y )
{
    return (y)? -log(hRes) : -log(1.0 - hRes);
    // return -y * log(hRes) - (1 - y) * (1 - log(hRes));
}

__global__ void GradientDescent(
    const unsigned int maxIter,
    const unsigned int alpha,
    const unsigned int numInstance,
    const unsigned int numFeatures,
    double* dBatchArr,
    double* dWeightArr,
    double* dDiffArr )
{
    unsigned int instanceId = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int featureId = threadIdx.y * blockDim.x + threadIdx.x;

    if (instanceId >= numInstance || featureId >= numFeatures) return;

    double costSumPre = 0.0;
    double deltaCostSum = 0.0;
    unsigned int iter = 0;

    if (featureId == 0) printf( "Instance ID: %u\n", instanceId );

    // do
    // {
    //     double costSumNew = 0.0;
    //     dBatchArr[featureId] = 0.0;
    //     // memset( dBatchArr, 0, numFeatures * sizeof( double ) );


    //     if (threadIdx == {0, 0, 0} && blockIdx == {0, 0, 0})
    //         printf( "Delta cost: %f\n", deltaCostSum );

    //     iter++;
    //     // __syncthreads();
    // }
    // while (iter == 1 || (deltaCostSum > 1.0 && iter < maxIter));
}

void cudaErrorCheck( cudaError_t cudaRes )
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

    unsigned int numInstance = trainSetImporter.GetNumInstances();
    double* featureBuff = trainSetImporter.GetInstances();
    unsigned short* classIndexBuff = trainSetImporter.GetClassIndex();
    vector<NumericAttr> featureVec = trainSetImporter.GetFeatures();
    unsigned int numFeatures = featureVec.size();

    // unsigned int numInstance = 25000;
    // unsigned int numFeatures = 1000;
    unsigned int alpha = 50;
    unsigned int maxIter = 1000;

    // Determine block and grid size
    dim3 blockStruct;
    dim3 gridStruct;
    if (numInstance < 1024) gridStruct.x = numInstance;
    else
    {
        gridStruct.x = 1024;
        gridStruct.y = (numInstance + gridStruct.x - 1) / gridStruct.x;
    }

    if (numFeatures < 1024) blockStruct.x = numFeatures;
    else
    {
        blockStruct.x = 1024;
        blockStruct.y = (numFeatures + blockStruct.x - 1) / blockStruct.x;
    }

    // normalize( featureVec, featureBuff, numInstance );

    // Node node = initNode( numFeatures );


    double* dBatchArr;
    double* dWeightArr;
    double* dDiffArr;
    double* dFeatureBuff;
    unsigned short* dClassBuff;
    cudaErrorCheck( cudaMalloc( (void**) &dBatchArr, numFeatures * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, numFeatures * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dDiffArr, numInstance * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureBuff, numInstance * numFeatures * sizeof( double ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassBuff, numInstance * sizeof( unsigned short ) ) );

    printf( "\nStart gradient descent...\n" );

    GradientDescent<<< gridStruct, blockStruct >>>(
        maxIter,
        alpha,
        numInstance,
        numFeatures,
        dBatchArr,
        dWeightArr,
        dDiffArr
    );

    // Gradient descent
    // do
    // {
    //     double costSumNew = 0.0;
    //     memset( batchArr, 0, numFeatures * sizeof( double ) );

    //     for (unsigned int i = 0; i < numInstance; i++)
    //     {
    //         // double hRes = activate( &node, &featureBuff[i * numFeatures] );

    //         double linearRes = weightBuff[numFeatures];
    //         for (unsigned int j = 0; j < numFeatures; j++)
    //             linearRes += weightBuff[j] * featureBuff[i * numFeatures + j];

    //         double hRes = 1.0 / (1.0 + exp(-linearRes));
    //         costSumNew += computeCost( hRes, classIndexBuff[i] );
    //         diffArr[i] = hRes - (double) classIndexBuff[i];
    //         // double diff = hRes - (double) classIndexBuff[i];
    //         // for (unsigned int j = 0; j < numFeatures; j++)
    //         //     batchArr[j] += diff * featureBuff[i * numFeatures + j];
    //     }

    //     for (unsigned int j = 0; j < numFeatures; j++)
    //     {
    //         batchArr[j] = 0;
    //         for (unsigned int i = 0; i < numInstance; i++)
    //             batchArr[j] += diffArr[i] * featureBuff[i * numFeatures + j];
    //         // Update weights
    //         weightBuff[j] -= alpha / (double) numInstance * batchArr[j];
    //     }

    //     deltaCostSum = costSumPre - costSumNew;
    //     costSumPre = costSumNew;

    //     printf( "Delta cost: %f\n", deltaCostSum );
    //     // printf( "Pre cost: %f\n", costSumPre );
    //     // printf( "New cost: %f\n", costSumNew );

    //     // Update weights
    //     // #pragma acc kernels loop
    //     // for (unsigned int j = 0; j < numFeatures; j++)
    //     //     weightBuff[j] -= alpha / (double) numInstance * batchArr[j];

    //     iter++;
    // }
    // while (iter == 1 || (deltaCostSum > 1.0 && iter < maxIter));

    cudaDeviceSynchronize();

    cudaFree(dBatchArr);
    cudaFree(dWeightArr);
    cudaFree(dDiffArr);

    return 0;
}
