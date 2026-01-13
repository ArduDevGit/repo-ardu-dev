//
// Created by gs on 1/12/26.
//

#ifndef REPO_ARDU_DEV_TRAININGDATAHELPER_H
#define REPO_ARDU_DEV_TRAININGDATAHELPER_H
#pragma once
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;


class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(VectorN<unsigned,NUM_LAYERS>& topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double>& inputVals);
    unsigned getTargetOutputs(vector<double>& targetOutputVals);

private:
    ifstream m_trainingDataFile;
};
#endif //REPO_ARDU_DEV_TRAININGDATAHELPER_H