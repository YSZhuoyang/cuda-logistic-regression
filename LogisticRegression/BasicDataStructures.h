
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_


using namespace std;

namespace BasicDataStructures
{
    struct Instance
    {
        double* featureAttrArray;
        unsigned short classIndex;
    };

    // Bound index of each instance with one of its feature value
    // and its class index, to minimize memory access time during
    // tree construction
    struct MiniInstance
    {
        double featureValue;
        unsigned int instanceIndex;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        double min;
        double max;
        double mean; // Not used.
    };

    struct Node
    {
        double* inputs;
        // Weight array has numFeatures + 1 elements.
        // The last element is bias parameter.
        double* weights;
        double output;
        double error;
        unsigned int numFeatures;
    };
}

#endif
