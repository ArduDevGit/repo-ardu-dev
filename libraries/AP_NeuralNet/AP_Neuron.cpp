

#include "AP_Neuron.h"
#include "../AP_HAL/AP_HAL.h"



const AP_HAL::HAL& hal = AP_HAL::get_HAL(); //TODO: change to hardware call

Neuron::Neuron() {
    activated = false;
    m_output = 0;
    m_myIndex = 0;

}

void Neuron::activate(unsigned numOutputs, unsigned myIndex) {

    activated = true;
    m_myIndex   = myIndex;
    uint8_t randomVal;

    // init/activate output to each node in next layer, with a random weight
    for (unsigned connection = 0; connection < numOutputs; ++connection) {
        m_outputWeights[connection].activated = true;
        hal.util->get_random_vals(&randomVal,1);
        m_outputWeights[connection].weight = randomVal / static_cast<float>(UINT32_MAX);
    }

}

void Neuron::feedForward(const Layer &prevLayer) {

    double sum = 0.0;

    // This neurons inputs =
    // sum of previous layers output nodes (include bias node) * the weight intended for this node
    for (unsigned n = 0; n < MAX_NEURONS; ++n) {
        if (prevLayer[n].activated) {
            sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
        }
    }

    m_output = /*trasnfer Funtion()*/sum;

}
