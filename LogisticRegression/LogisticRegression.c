
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

inline double activate( Node* node, double* inputArr )
{
    double linearRes = node->weights[node->numFeatures];
    node->inputs = inputArr;

    unsigned int numFeatures = node->numFeatures;
    for (unsigned int i = 0; i < numFeatures; i++)
        linearRes += node->weights[i] * node->inputs[i];

    node->output = 1.0 / (1.0 + exp(-linearRes));

    // printf( "lres: %f\n", linearRes );
    
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
    Instance* instTable = trainSetImporter.GetInstances();
    vector<NumericAttr> featureVec = trainSetImporter.GetFeatures();
    unsigned int numFeatures = featureVec.size();

    Node node = initNode( numFeatures );
    unsigned int iter = 0;
    unsigned int maxIter = 1;
    double costSum = 0.0;
    double alpha = 1.0;
    double* batchArr = (double*) malloc( numFeatures * sizeof( double ) );

    // Gradient descent
    do
    {
        costSum = 0.0;
        memset( batchArr, 0, numFeatures * sizeof( double ) );

        for (unsigned int i = 0; i < numInst; i++)
        {
            double hRes = activate( &node, instTable[i].featureAttrArray );
            double cost = computeCost( hRes, instTable[i].classIndex );
            costSum += cost;

            // printf( "hres: %f\n", hRes );

            for (unsigned int j = 0; j < numFeatures; j++)
                batchArr[j] += cost * node.inputs[j];
        }

        printf( "cost: %f\n", costSum );

        // UpdateWeights
        for (unsigned int j = 0; j < numFeatures; j++)
        {
            printf( "test delta %f\n", batchArr[j] );
            node.weights[j] -= alpha / (double) numInst * batchArr[j];
        }
    }
    while (costSum > 1.0 && iter++ < maxIter);
    
    return 0;
}
