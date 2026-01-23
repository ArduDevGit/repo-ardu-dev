

#include "AP_Neural_Net.h"

#include <cassert>
#include <iostream> //TODO: change to console
#include <cmath>

Net::Net(const VectorN<unsigned,NUM_LAYERS> &topology) : m_topologyNumNeurons(topology)
{

    for (unsigned layerNum = 0; layerNum < NUM_LAYERS; ++layerNum) {

        // grab number of outputs from the "layer forward"
        unsigned numOutputs = layerNum == (NUM_LAYERS-1) ? 0 : topology[layerNum+1];

        // create/activate neurons in this layer
        // fill (activate) each layer with the number of neurons specified by topology
        for (unsigned neuroNum = 0; neuroNum < topology[layerNum]; ++neuroNum) {
            assert(topology[layerNum] <= MAX_NEURONS);
            // pass in number of outputs the neuron will need to feed the next layer (except the output/last layer).
            m_layers[layerNum][neuroNum].activate(numOutputs, neuroNum);
        }

        // Force last neuron for this iteration/layer to 1.0 - the bias neuron
        unsigned neuronsInLayer = topology[layerNum];
        bool isOutputLayer = (layerNum == (NUM_LAYERS-1));
        if (!isOutputLayer && neuronsInLayer > 0) {
            m_layers[layerNum][neuronsInLayer-1].setOutputVal(1.0);
        }
    }


    displayActiveNeurons(); //TODO: remove debug
    // init variables
    m_error = 0.0;
    m_recentAverageError = 0.0;
    m_recentAverageSmoothingFactor = 0.0;

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
        unsigned numActivePrevLayer = m_topologyNumNeurons[layerNum-1];

        // Loop through number of neurons active in this layer
        for (unsigned n = 0; n < m_topologyNumNeurons[layerNum]; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer,numActivePrevLayer); // give a reference to previous layer only
        }
    }
}

// The learning algorithm for the neural net.
// It adjusts the weights of the connections between neurons
// so that the network’s outputs get closer to the desired targets.
void Net::backPropagate(const VectorN<float,NUM_NEURONS_OUTPUT_LAYER> &targets)
{
    // calculate overall net error (Root Mean Square Error - RMS), compare current outputs to target given
    Layer &outputLayer = m_layers[NUM_LAYERS-1];
    m_error = 0.0; // and accumulate overall net error
    // loop through output layer neurons only (not including bias)
    for (unsigned neuron = 0; neuron < NUM_NEURONS_OUTPUT_LAYER; ++neuron) {
        float delta = targets[neuron] - outputLayer[neuron].getOutputVal();
        m_error += delta * delta; // sum of squares of errors
    }

    // get average error (divide by number of elements that we summed for the average)
    m_error = m_error / (NUM_NEURONS_OUTPUT_LAYER); // exclude bias neuron
    m_error = std::sqrt(m_error);

    // debug: recent average measurement:
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // calculate output layer gradients for each output neuron,
    // meaning compute how much that neuron contributed to the overall error.
    for (unsigned neuron = 0; neuron < NUM_NEURONS_OUTPUT_LAYER; ++neuron) {
        outputLayer[neuron].calcOutputGradients(targets[neuron]);
    }

    // calculate how much hidden layer neurons contributed to the overall error.
    for (unsigned layerNum = NUM_LAYERS - 2; layerNum > 0; --layerNum) {
        Layer &currentHiddenLayer = m_layers[layerNum];
        Layer &nextHiddenLayer = m_layers[layerNum+1];
        unsigned numActiveNextLayer = m_topologyNumNeurons[layerNum+1];

        for (unsigned neuron = 0; neuron < m_topologyNumNeurons[layerNum] - 1; ++neuron) {
                currentHiddenLayer[neuron].calcHiddenGradient(nextHiddenLayer,numActiveNextLayer);
        }
    }

    // for all layers from outputs to first hidden layer, update connection weights
    for (unsigned layerNum = NUM_LAYERS - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum-1];
        unsigned numActivePrevLayer = m_topologyNumNeurons[layerNum-1];

        bool isOutputLayer = (layerNum == NUM_LAYERS - 1);
        unsigned limit = isOutputLayer ?
                         m_topologyNumNeurons[layerNum] :     // update ALL neurons
                         m_topologyNumNeurons[layerNum] - 1;  // skip bias neuron

        for (unsigned neuron = 0; neuron < limit; ++neuron) {
            layer[neuron].updateInputWeights(prevLayer, numActivePrevLayer);
        }
    }


}

void Net::getResults(VectorN<float,NUM_NEURONS_OUTPUT_LAYER> &results) const
{
    for (unsigned n = 0; n < NUM_NEURONS_OUTPUT_LAYER; ++n) {
        results[n] = m_layers[NUM_LAYERS-1][n].getOutputVal();
    }
}

//TODO: send to console
void Net::displayActiveNeurons() const {

    std::cout << "ACTIVE NEURONS: " << std::endl;
    for (unsigned layerNum = 0; layerNum < NUM_LAYERS; ++layerNum) {
        for (unsigned neuroNum = 0; neuroNum < m_topologyNumNeurons[layerNum]; ++neuroNum) {
                std::cout << "  LAYER: " << layerNum << " NEURON: " << neuroNum << " ACTIVATED" << std::endl;
        }
    }

}