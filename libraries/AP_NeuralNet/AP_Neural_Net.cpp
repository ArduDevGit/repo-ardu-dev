

#include "AP_Neural_Net.h"

#include <cassert>
#include <iostream> //TODO: change to console
#include <cmath>

Net::Net(const VectorN<unsigned,NUM_LAYERS> &topology) : m_topology(topology)
{

    for (unsigned layerNum = 0; layerNum < NUM_LAYERS; ++layerNum) {

        // number of outputs a neuron will need to feed the next layer (except for the output/last layer).
        unsigned numOutputs = layerNum == (NUM_LAYERS-1) ? 0 : topology[layerNum+1];

        // fill (activate) each layer with the number of neurons specified
        for (unsigned neuroNum = 0; neuroNum < topology[layerNum]; ++neuroNum) {
            assert(topology[layerNum] <= MAX_NEURONS);
            m_layers[layerNum][neuroNum].activate(numOutputs, neuroNum);
        }
    }

    displayActiveNeurons(); //TODO: remove debug

}

void Net::feedForward(const VectorN<float,NUM_INPUTS> &inputs)
{
    // assign input values into input layer neurons
    for (unsigned i = 0; i < NUM_INPUTS; ++i) {
        m_layers[0][i].setOutputVal(inputs[i]);
    }

    // forward propagate after input layer
    for (unsigned layerNum = 1; layerNum < NUM_LAYERS; ++layerNum) {

        Layer &prevLayer = m_layers[layerNum-1];

        for (unsigned n = 0; n < MAX_NEURONS; ++n) {
            if (m_layers[layerNum][n].isActivated()) {
                m_layers[layerNum][n].feedForward(prevLayer); // give a reference to previous layer only
            }
        }
    }
}

// The learning algorithm for the neural net.
// It adjusts the weights of the connections between neurons
// so that the network’s outputs get closer to the desired targets.
void Net::backPropagate(const VectorN<float,NUM_INPUTS> &targets)
{
    // calculate overall net error (Root Mean Square Error - RMS), compare current outputs to target given

    Layer &outputLayer = m_layers[NUM_LAYERS-1];
    m_error = 0.0; // and accumulate overall net error
    // loop through output layer neurons
    for (unsigned neuron = 0; neuron < NUM_NEURONS_OUTPUT_LAYER - 1; ++neuron) {
        float delta = targets[neuron] - outputLayer[neuron].getOutputVal();
        m_error += delta * delta; // sum of squares of errors
    }

    // get average error (divide by number of elements that we summed for the average)
    m_error = m_error / NUM_NEURONS_OUTPUT_LAYER - 1;
    m_error = std::sqrt(m_error);



    // calculate output layer gradients for each output neuron (compute how much its output contributed to the error).

    // calculate gradients on hidden layers

    // for all layers from outputs to first hidden layer, update connection weights

}

void Net::getResults(VectorN<float,NUM_INPUTS> &results) const
{

}

//TODO: send to console
void Net::displayActiveNeurons() const {

    std::cout << "ACTIVE NEURONS: " << std::endl;
    for (unsigned layerNum = 0; layerNum < NUM_LAYERS; ++layerNum) {
        for (unsigned neuroNum = 0; neuroNum < m_topology[layerNum]; ++neuroNum) {
            if (m_layers[layerNum][neuroNum].isActivated()) {
                std::cout << "  LAYER: " << layerNum << " NEURON: " << neuroNum << " ACTIVATED" << std::endl;
            }
        }
    }

}