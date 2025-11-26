


// This is a very simple feed forward neural net.
// Adapted to an embedded (ArduPilot) system, from Dave Miller's neural net in C++:
// https://millermattson.com/dave/?p=54
//
// Notes:
// Each colum of neurons is a layer. Need at least 1 input layer (to hold input), 1 hidden layer and 1 output layer.
// Hidden layer(s): where each neuron connects forward, to each neuron in the next forward layer.
// Each neuron does some function with the inputs while weighting them (weights are constantly changing) and creates
// an output at that neuron, and fed forward.
// Bias neurons: 1 bias neuron with a fixed value is added at each layer to feed into the next.

#pragma once

//https://vimeo.com/19569529
//58min

#include "AP_Neuron.h"
#include <AP_Math/vectorN.h>




class Net {

public:
    explicit Net(const VectorN<unsigned,NUM_LAYERS> &topology);

    void feedForward(const VectorN<float,NUM_INPUTS> &inputs);
    void backPropagate(const VectorN<float,NUM_INPUTS> &targets);
    void getResults(VectorN<float,NUM_INPUTS> &results) const;



private:
    VectorN<Layer, NUM_LAYERS> m_layers; // i.e. vector of vector [layer][neuronNum]
    void displayActiveNeurons() const;

    VectorN<unsigned,NUM_LAYERS> m_topology;

    float m_error;

    float m_recentAverageError;
    float m_recentAverageSmoothingFactor;

};

