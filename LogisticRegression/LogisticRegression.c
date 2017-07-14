#include "Helper.h"
#include "ArffImporter.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>



Node initNode( unsigned int numFeatures )
{
    Node node;
    node.numFeatures = numFeatures;
    node.weights = (float*) malloc( (numFeatures + 1) * sizeof( float ) );
    memset( node.weights, 1, (numFeatures + 1) * sizeof( float ) );

    return node;
}

void normalize(
    std::vector<NumericAttr> featureVec,
    float* featureBuff,
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
            featureBuff[featureIndex] =
                (featureBuff[featureIndex] - featureVec[i].mean) / range;
        }
    }
}

inline float activate(
    Node* node,
    float* inputArr )
{
    float linearRes = node->weights[node->numFeatures];
    node->inputs = inputArr;

    unsigned int numFeatures = node->numFeatures;
    for (unsigned int i = 0; i < numFeatures; i++)
        linearRes += node->weights[i] * node->inputs[i];

    node->output = 1.0 / (1.0 + exp(-linearRes));

    return node->output;
}

inline float computeCost( float hRes, unsigned short y )
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
    float* featureBuff = trainSetImporter.GetFeatureBuff();
    unsigned short* classIndexBuff = trainSetImporter.GetClassIndex();
    std::vector<NumericAttr> featureVec = trainSetImporter.GetFeatures();
    unsigned int numFeatures = featureVec.size();

    normalize( featureVec, featureBuff, numInst );

    Node node = initNode( numFeatures );
    unsigned int iter = 0;
    unsigned int maxIter = 200;
    float costSumPre = 0.0;
    float deltaCostSum = 0.0;
    float alpha = 50.0;
    float* batchArr = (float*) malloc( numFeatures * sizeof( float ) );

    time_t start, end;
    double dif;
    time( &start );
    
    // Gradient descent
    do
    {
        float costSumNew = 0.0;
        memset( batchArr, 0, numFeatures * sizeof( float ) );

        for (unsigned int i = 0; i < numInst; i++)
        {
            float hRes = activate( &node, &featureBuff[i * numFeatures] );
            float diff = hRes - (float) classIndexBuff[i];
            costSumNew += computeCost( hRes, classIndexBuff[i] );
            for (unsigned int j = 0; j < numFeatures; j++)
                batchArr[j] += diff * node.inputs[j];
        }

        deltaCostSum = costSumPre - costSumNew;
        costSumPre = costSumNew;

        // printf( "Delta cost: %f\n", deltaCostSum );

        // Update weights
        printf( "Weight: %f\n", node.weights[0] );
        for (unsigned int j = 0; j < numFeatures; j++)
            node.weights[j] -= alpha / (float) numInst * batchArr[j];
        
        iter++;
    }
    while (iter == 1 || (deltaCostSum > 1.0 && iter < maxIter));

    time( &end );
    dif = difftime( end, start );

    printf( "Time taken is %.2lf seconds.\n", dif );
    
    return 0;
}
