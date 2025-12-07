

#include <iostream>
#include <AP_gtest.h>
#include <AP_NeuralNet/AP_Neural_Net.h>
#include <cmath>
#include <cstdlib>
#include <fstream>




using namespace std;

TEST(AP_NEURAL_NET, nn_generate_data)
{
    cout << "random training sets for XOR -- two inputs and one output" << endl;

    // Open a file for writing
    ofstream myfile("/home/gs/CLionProjects/repo-ardu-dev/libraries/AP_NeuralNet/tests/trainingData.txt");

    myfile << "topology: 4 3 2" << endl;
    for (int i = 2000; i >= 0; --i) {
        int n1 = static_cast<int>(2.0 * rand() / static_cast<double>(RAND_MAX));
        int n2 = static_cast<int>(2.0 * rand() / static_cast<double>(RAND_MAX));
        int t = n1 ^ n2; // should be 0 or 1
        myfile << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
        myfile << "out: " << t << ".0" << endl;
    }
    // Close the file
    myfile.close();
}


TEST(AP_NEURAL_NET, nn_test)
{
    cout << "START NET TEST" << endl;

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
    VectorN<float,NUM_NEURONS_OUTPUT_LAYER> targets;
    nNet.backPropagate(targets);

    // operation after training
    // feed forward real data
    // TBD
    //get results
    VectorN<float, NUM_NEURONS_OUTPUT_LAYER> results {};
    nNet.getResults(results);


    cout << "END NET TEST" << endl;
}

AP_GTEST_MAIN()