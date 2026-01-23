#ifndef REPO_ARDU_DEV_TRAININGDATAHELPER_H
#define REPO_ARDU_DEV_TRAININGDATAHELPER_H
#pragma once
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <AP_Math/vectorN.h>

#include "AP_Neural_Config.h"

using namespace std;


class TrainingData
{
public:
    TrainingData();
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(VectorN<unsigned,NUM_LAYERS>& topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(VectorN<float,NUM_INPUTS> & inputVals);
    unsigned getTargetOutputs(VectorN<float,NUM_NEURONS_OUTPUT_LAYER>& targetOutputVals);

private:
    ifstream m_trainingDataFile;
};
#endif //REPO_ARDU_DEV_TRAININGDATAHELPER_H