

#include "AP_Neuron.h"
#include "../AP_HAL/AP_HAL.h"

#include <cmath>


const AP_HAL::HAL& hal = AP_HAL::get_HAL(); //TODO: change to hardware call

// Tuning parameters - these are consistent for all neurons
float Neuron::eta = 0.15; // overall net learning rate [0.0,1.0]
float Neuron::alpha = 0.5; // momentum - multiplier of the last weight change [0.0,n]




Neuron::Neuron() {
    activated = false;
    m_output = 0;
    m_myIndex = 0;
    m_gradient = 0.0;

}

void Neuron::activate(unsigned numOutputs, unsigned myIndex) {

    activated = true;
    m_myIndex   = myIndex;
    uint8_t randomVal;

    // init/activate output to each node in next layer, with a random weight
    for (unsigned connection = 0; connection < numOutputs; ++connection) {
        m_outputWeights[connection].activated = true;
        m_outputWeights[connection].deltaWeight = 0.0;

        hal.util->get_random_vals(&randomVal,1);
        // scale weight between [-0.5, +0.5]
        //m_outputWeights[connection].weight = (randomVal / static_cast<float>(UINT8_MAX)) - 0.5f;
        m_outputWeights[connection].weight = (randomVal / 255.0f) * 2.0f - 1.0f;  // [-1.0, +1.0]

    }

}

void Neuron::feedForward(const Layer &prevLayer, unsigned numActivePrevLayer) {

    float sum = 0.0;

    // This neurons inputs =
    // sum of previous layers output nodes (include bias node) * the weight intended for this node
    for (unsigned n = 0; n < numActivePrevLayer; ++n) {
            sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_output = Neuron::transferFunction(sum);

}

// will need to reevaluate transfer functions to see if will still work for turbulence detection input.
// may want ReLU or leaky ReLU.
float Neuron::transferFunction(float sum) {
    // using hyperbolic tangent function scaled to [-1.0...1.0]
    // using from math.h for now, can convert to approximation routine as well if needed for performance
    return std::tanh(sum);
}

float Neuron::transferFunctionDerivative(float output) {
    // tanh derivitive - d/dx of tanh(x)
    // using approximation here
    return 1.0 - output * output;
}

// gradient a.k.a. error signal
void Neuron::calcOutputGradients(float targetVal) {
    // how far off was this neurons output
    float delta = targetVal - m_output;

    // the final "blame score" for this neuron (error size * "sensitivity")
    // The gradient is saying this neuron is off by this much,
    // and here’s how strongly we should adjust weights.
    m_gradient = delta * Neuron::transferFunctionDerivative(m_output);
}

// sum of errors weighted by connection strength, from the errors in the next layer.
float Neuron::sumDOW(const Layer &nextHiddenLayer, unsigned numActiveNextLayer) const {
    float sum = 0.0;

    for (unsigned n = 0; n < numActiveNextLayer - 1; ++n) { // exclude bias neuron
            // sum the weight from our neuron to the other neuron we feed
            sum += m_outputWeights[n].weight * nextHiddenLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradient(const Layer &nextHiddenLayer, unsigned numActiveNextLayer) {

    float dow = sumDOW(nextHiddenLayer,numActiveNextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_output);
}

void Neuron::updateInputWeights(Layer &prevLayer, unsigned numActivePrevLayer) {

    for (unsigned n = 0; n < numActivePrevLayer; ++n) {
            Neuron &neuron = prevLayer[n];
            float oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

            float newDeltaWeight =
                // learning rate * prev neuron output * our neurons gradient + (momentum * old change in weight)
                eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

            // update the neurons weight it is storing for me (this neuron)
            neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
            neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}