
#pragma once

#include "AP_Neural_Config.h"
#include <AP_Math/vectorN.h>


class Neuron;
// define a layer as a vector of neurons
typedef VectorN<Neuron,MAX_NEURONS> Layer;


struct Connection {
    float weight;
    float deltaWeight;
    bool activated;
};


class Neuron {

public:
    Neuron();
    void activate(unsigned numOutputs, unsigned myIndex);
    void deactivate() { activated = false; }
    bool isActivated() const { return activated; }

    void setOutputVal(float val) { m_output = val; }
    float getOutputVal() const { return m_output; }

    void feedForward(const Layer &prevLayer);

private:
    bool activated;
    float m_output;
    unsigned m_myIndex;

    VectorN<Connection, MAX_NEURONS> m_outputWeights;

    static double transferFunction(double sum);
    static double transferFunctionDerivative(double sum);

};


