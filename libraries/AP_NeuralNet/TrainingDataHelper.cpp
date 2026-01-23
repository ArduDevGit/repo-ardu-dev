#include "TrainingDataHelper.h"


void TrainingData::getTopology(VectorN<unsigned,NUM_LAYERS>& topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    unsigned index = 0;
    unsigned n;

    while (ss >> n && index < NUM_LAYERS) {
        //topology.push_back(n);
        topology[index] = n;
        index++;
    }
}

TrainingData::TrainingData()
{
    const string filename = "libraries/AP_NeuralNet/tests/trainingData.txt";
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(VectorN<float,NUM_INPUTS> & inputVals)
{
    //inputVals.clear();
    // clear out container for next input:
    for (unsigned i = 0; i < NUM_INPUTS; i++) {
        inputVals[i] = 0.0f;
    }

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    unsigned index = 0;

    if (label.compare("in:") == 0) {
        float oneValue;
        while (ss >> oneValue && index < NUM_INPUTS) {
            //inputVals.push_back(oneValue);
            inputVals[index] = oneValue;
            index++;
        }
    }

    //return inputVals.size();
    return index;
}

unsigned TrainingData::getTargetOutputs(VectorN<float,NUM_NEURONS_OUTPUT_LAYER>& targetOutputVals)
{
    //targetOutputVals.clear();
    // clear container before using
    for (unsigned i = 0; i < NUM_NEURONS_OUTPUT_LAYER; i++) {
        targetOutputVals[i] = 0.0f;
    }

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    unsigned index = 0;

    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue && index < NUM_NEURONS_OUTPUT_LAYER) {
            targetOutputVals[index] = oneValue;
            index++;
            //targetOutputVals.push_back(oneValue);
        }
    }

    //return targetOutputVals.size();
    return index;
}