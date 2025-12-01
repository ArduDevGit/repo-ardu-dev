
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

    void calcOutputGradients(float targetVal);
    void calcHiddenGradient(const Layer &nextHiddenLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    bool activated;
    float m_output;
    unsigned m_myIndex;
    float m_gradient;

    // eta and aplha are consistent for all neurons
    static float eta; // overall net learning rate [0.0,1.0]
    static float alpha; // momentum - multiplier of the last weight change [0.0,n]

    VectorN<Connection, MAX_NEURONS> m_outputWeights;

    static float transferFunction(float sum);
    static float transferFunctionDerivative(float output);

    float sumDOW(const Layer &nextHiddenLayer) const;

};


