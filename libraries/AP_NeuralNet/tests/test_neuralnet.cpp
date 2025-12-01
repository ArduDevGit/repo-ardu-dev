

#include <iostream>
#include <AP_gtest.h>
#include <AP_NeuralNet/AP_Neural_Net.h>



TEST(AP_NEURAL_NET, nn_test)
{
    std::cout << "START NET TEST" << std::endl;

    // number of layers and number of neurons in each layer,
    // i.e. 3-2-1  input layer has 3 neurons, 1 hidden has 2 neurons and output layer has 1 neuron.
    VectorN<unsigned,NUM_LAYERS> topology;
    topology[0] = NUM_NEURONS_INPUT_LAYER;
    topology[1] = NUM_NEURONS_LAYER_ONE;
    topology[2] = NUM_NEURONS_OUTPUT_LAYER;
    Net nNet(topology);


    // input training data
    VectorN<float,NUM_INPUTS> inputs;
    nNet.feedForward(inputs);

    // tell it the answers for backpropagation
    VectorN<float,NUM_INPUTS> targets;
    nNet.backPropagate(targets);

    // operation after training
    // feed forward real data
    // TBD
    //get results
    VectorN<float, NUM_NEURONS_OUTPUT_LAYER> results {};
    nNet.getResults(results);


    std::cout << "END NET TEST" << std::endl;
}

AP_GTEST_MAIN()