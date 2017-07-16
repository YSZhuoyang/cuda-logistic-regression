#include "Helper.h"
#include "ArffImporter.h"

#include <cublas_v2.h>


#define MAX_ITER      200
#define LEARNING_RATE 50.0f

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
    const float* __restrict__ dFeaCostProdArr,
    const float updateWParam,
    const unsigned int numInstances,
    const unsigned int numFeatures )
{
    unsigned int featureId = blockIdx.x * blockDim.x + threadIdx.x;
    if (featureId >= numFeatures) return;

    dWeightArr[featureId] -= updateWParam * dFeaCostProdArr[featureId];

    if (featureId == 0)
        printf( "Updating weights completed, weight: %f\n", dWeightArr[0] );
}

inline void cudaErrorCheck( cudaError_t cudaStatus )
{
    if (cudaStatus != cudaSuccess)
        printf(
            "kernel launch failed with error \"%s\".\n",
            cudaGetErrorString( cudaStatus )
        );
}

inline void cublasErrorCheck( cublasStatus_t cublasStatus )
{
    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        printf( "CuBLAS launch failed with error\n" );
        switch (cublasStatus)
        {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                printf( "CUBLAS_STATUS_NOT_INITIALIZED\n" );

            case CUBLAS_STATUS_ALLOC_FAILED:
                printf( "CUBLAS_STATUS_ALLOC_FAILED\n" );

            case CUBLAS_STATUS_INVALID_VALUE:
                printf( "CUBLAS_STATUS_INVALID_VALUE\n" );

            case CUBLAS_STATUS_ARCH_MISMATCH:
                printf( "CUBLAS_STATUS_ARCH_MISMATCH\n" );

            case CUBLAS_STATUS_MAPPING_ERROR:
                printf( "CUBLAS_STATUS_MAPPING_ERROR\n" );

            case CUBLAS_STATUS_EXECUTION_FAILED:
                printf( "CUBLAS_STATUS_EXECUTION_FAILED\n" );

            case CUBLAS_STATUS_INTERNAL_ERROR:
                printf( "CUBLAS_STATUS_INTERNAL_ERROR\n" );
        }
    }
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

    /* Determine block and grid size of UpdateWeight kernel */
    dim3 uwBlockDim;
    dim3 uwGridDim;
    if (numFeatures > 1024)
    {
        uwBlockDim.x = 1024;
        uwGridDim.x = (numInstances + 1023) / 1024;
    }
    else uwBlockDim.x = numFeatures;

    /* Determine block and grid size of ComputeCost kernel */
    dim3 ccBlockDim;
    dim3 ccGridDim;
    if (numInstances > 1024)
    {
        ccBlockDim.x = 1024;
        ccGridDim.x = (numInstances + 1023) / 1024;
    }
    else ccBlockDim.x = numInstances;

    // Init CuBLAS
    cublasHandle_t cublasHandle;
    cublasErrorCheck( cublasCreate( &cublasHandle ) );

    // Init host data
    float* biasArr = (float*) malloc( numInstances * sizeof( float ) );
    for (unsigned int i = 0; i < numInstances; i++)
        biasArr[i] = node.weights[0];

    // Init device data
    float* dBiasArr = nullptr;
    float* dCostArr = nullptr;
    float* dWeightArr = nullptr;
    float* dFeatureMat = nullptr;
    float* dFeatureMatTrans = nullptr;
    float* dFeaCostProdArr = nullptr;
    unsigned short* dClassArr = nullptr;
    // One feature per row, one instance per column
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMat, numInstances * numFeatures * sizeof( float ) ) );
    // One instance per row, one feature per column
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMatTrans, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dCostArr, numInstances * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dBiasArr, numInstances * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassArr, numInstances * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeaCostProdArr, numFeatures * sizeof( float ) ) );
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
        // First element is used as bias, start from the second one
        node.weights + 1,
        numFeatures * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassArr,
        classArr,
        numInstances * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dBiasArr,
        biasArr,
        numInstances * sizeof( float ),
        cudaMemcpyHostToDevice ) );

    // Gradient descent params
    float updateWParam = LEARNING_RATE / (float) numInstances;
    unsigned int iter = 0;

    time_t start, end;
    float dif;
    time( &start );
    
    printf( "\nStart gradient descent...\n" );

    float alpha = 1.0f;
    float preBeta = 1.0f;
    float uwBeta = 0.0f;
    // Gradient descent
    while (iter++ < MAX_ITER)
    {
        // Reset dCostArr with Bias
        cudaErrorCheck( cudaMemcpyAsync(
            dCostArr,
            dBiasArr,
            numInstances * sizeof( float ),
            cudaMemcpyDeviceToDevice ) );
        // Predict
        cublasErrorCheck( cublasSgemv(
            cublasHandle,
            CUBLAS_OP_N,
            numInstances,
            numFeatures,
            &alpha,
            dFeatureMatTrans,
            numInstances,
            dWeightArr,
            1,
            &preBeta,
            dCostArr,
            1 ) );

        ComputeCost<<< ccGridDim, ccBlockDim >>>(
            dCostArr,
            dClassArr,
            numInstances );
        cudaErrorCheck( cudaGetLastError() );

        // Cost vec dot FeaMat
        cublasErrorCheck( cublasSgemv(
            cublasHandle,
            CUBLAS_OP_N,
            numFeatures,
            numInstances,
            &alpha,
            dFeatureMat,
            numFeatures,
            dCostArr,
            1,
            &uwBeta,
            dFeaCostProdArr,
            1 ) );
        UpdateWeight<<< uwGridDim, uwBlockDim >>>(
            dWeightArr,
            dFeaCostProdArr,
            updateWParam,
            numInstances,
            numFeatures );
        cudaErrorCheck( cudaGetLastError() );
    }
    cudaErrorCheck( cudaThreadSynchronize() );

    cublasErrorCheck( cublasDestroy( cublasHandle ) );
    cudaMemcpy(
        node.weights + 1,
        dWeightArr,
        numFeatures * sizeof( float ),
        cudaMemcpyDeviceToHost );

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken is %.2lf seconds.\n", dif );

    cudaFree( dFeatureMat );
    cudaFree( dFeatureMatTrans );
    cudaFree( dClassArr );
    cudaFree( dWeightArr );
    cudaFree( dCostArr );
    cudaFree( dFeaCostProdArr );
    cudaFree( dBiasArr );
    free( node.weights );
    free( biasArr );

    return 0;
}
