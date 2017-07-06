
#include "Helper.h"
#include "ArffImporter.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>



Node initNode( unsigned int numFeatures )
{
    Node node;
    node.numFeatures = numFeatures;
    node.weights = (double*) malloc( (numFeatures + 1) * sizeof( double ) );
    memset( node.weights, 1, (numFeatures + 1) * sizeof( double ) );

    return node;
}

void normalize(
    vector<NumericAttr> featureVec,
    double* featureBuff,
    unsigned int numInst )
{
    unsigned int numFeatures = featureVec.size();

    for (unsigned int i = 0; i < numFeatures; i++)
    {
        // Use either range / standard deviation
        double range = featureVec[i].max - featureVec[i].min;
        if (range == 0.0) continue;

        for (unsigned int j = 0; j < numInst; j++)
        {
            unsigned int featureIndex = j * numFeatures + i;
            featureBuff[featureIndex] =
                (featureBuff[featureIndex] - featureVec[i].mean) / range;
        }
    }
}

inline double activate(
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

inline double computeCost( double hRes, unsigned short y )
{
    return (y)? -log(hRes) : -log(1.0 - hRes);
    // return -y * log(hRes) - (1 - y) * (1 - log(hRes));
}

int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    unsigned int numInst = trainSetImporter.GetNumInstances();
    double* featureBuff = trainSetImporter.GetInstances();
    unsigned short* classIndexBuff = trainSetImporter.GetClassIndex();
    vector<NumericAttr> featureVec = trainSetImporter.GetFeatures();
    unsigned int numFeatures = featureVec.size();

    normalize( featureVec, featureBuff, numInst );

    Node node = initNode( numFeatures );
    unsigned int iter = 0;
    unsigned int maxIter = 2000;
    double costSumPre = 0.0;
    double deltaCostSum = 0.0;
    double alpha = 50.0;
    double* batchArr = (double*) malloc( numFeatures * sizeof( double ) );
    double* diffArr = (double*) malloc( numInst * sizeof( double ) );

    // Array copied into video memo
    double* weightBuff = node.weights;

    // Gradient descent
    #pragma acc data copy(weightBuff[:numFeatures + 1]) copyin(featureBuff[:numInst * numFeatures], classIndexBuff[:numInst]) create(batchArr[:numFeatures], diffArr[:numInst])
    do
    {
        double costSumNew = 0.0;
        // memset( batchArr, 0, numFeatures * sizeof( double ) );

        #pragma acc kernels
        {
            #pragma acc loop
            for (int i = 0; i < numFeatures; i++)
                batchArr[i] = 0;

            #pragma acc loop
            for (unsigned int i = 0; i < numInst; i++)
            {
                // double hRes = activate( &node, &featureBuff[i * numFeatures] );

                double linearRes = weightBuff[numFeatures];
                for (unsigned int j = 0; j < numFeatures; j++)
                    linearRes += weightBuff[j] * featureBuff[i * numFeatures + j];

                double hRes = 1.0 / (1.0 + exp(-linearRes));
                costSumNew += computeCost( hRes, classIndexBuff[i] );
                diffArr[i] = hRes - (double) classIndexBuff[i];
                // double diff = hRes - (double) classIndexBuff[i];
                // for (unsigned int j = 0; j < numFeatures; j++)
                //     #pragma acc atomic
                //     batchArr[j] += diff * featureBuff[i * numFeatures + j];
            }

            #pragma acc loop
            for (unsigned int j = 0; j < numFeatures; j++)
            {
                for (unsigned int i = 0; i < numInst; i++)
                    batchArr[j] += diffArr[i] * featureBuff[i * numFeatures + j];
                // Update weights
                weightBuff[j] -= alpha / (double) numInst * batchArr[j];
            }
        }

        deltaCostSum = costSumPre - costSumNew;
        costSumPre = costSumNew;

        printf( "Delta cost: %f\n", deltaCostSum );
        // printf( "Pre cost: %f\n", costSumPre );
        // printf( "New cost: %f\n", costSumNew );

        // Update weights
        // #pragma acc kernels loop
        // for (unsigned int j = 0; j < numFeatures; j++)
        //     weightBuff[j] -= alpha / (double) numInst * batchArr[j];

        iter++;
    }
    while (iter == 1 || (deltaCostSum > 1.0 && iter < maxIter));

    free( batchArr );
    free( diffArr );

    return 0;
}
