


#include <AP_gtest.h>
#include <AP_NeuralNet/AP_Neural_Net.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../TrainingDataHelper.h"


using namespace std;



TEST(AP_NEURAL_NET, nn_generate_data)
{
    cout << "Random training data for XOR (two inputs and one output)" << endl;

    ofstream myfile("libraries/AP_NeuralNet/tests/trainingData.txt");

    myfile << "topology: 3 3 1" << endl;

    srand(time(NULL));

    for (int i = 0; i < 2000; ++i) {
        int n1 = rand() % 2;
        int n2 = rand() % 2;

        // XOR target in [-1, +1]
        int xorVal = n1 ^ n2;
        float t = xorVal ? 1.0f : -1.0f;

        myfile << "in: " << n1 << ".0 " << n2 << ".0" << endl;
        myfile << "out: " << t << endl;
    }

    myfile.close();
}



void showVectorVals(string label, VectorN<float,NUM_NEURONS_OUTPUT_LAYER>& v);
void showVectorValsInputs(string label, VectorN<float,NUM_INPUTS>& v);

TEST(AP_NEURAL_NET, nn_test)
{
    cout << "START NET TEST" << endl;

    TrainingData trainData;


    // set the topology
    VectorN<unsigned,NUM_LAYERS> topology;
    trainData.getTopology(topology);

    //setup net structure with the topology
    Net myNet(topology);

    VectorN<float,NUM_INPUTS> inputVals;
    VectorN<float,NUM_NEURONS_OUTPUT_LAYER> resultVals;
    VectorN<float,NUM_NEURONS_OUTPUT_LAYER> targetvals;
    int trainingPass = 0;

    while (!trainData.isEof())
    {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // get new input data and feet it forward
        if (trainData.getNextInputs(inputVals) != (topology[0] - 1))
        {
            break;
        }
        showVectorValsInputs(": inputs:", inputVals);
        myNet.feedForward(inputVals);

        // collect the nets actual results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // train the net what the outputs should have been
        trainData.getTargetOutputs(targetvals);
        showVectorVals("Targets:", targetvals);
        //assert(targetvals.size() == topology.back());

        myNet.backPropagate(targetvals);

        //report how well the training is working, averaged
        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;

    }

    cout << endl << "END NET TEST" << endl;
}

void showVectorVals(string label, VectorN<float,NUM_NEURONS_OUTPUT_LAYER>& v)
{
    cout << label << " ";
    for (unsigned i = 0; i < NUM_NEURONS_OUTPUT_LAYER; ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

void showVectorValsInputs(string label, VectorN<float,NUM_INPUTS>& v)
{
    cout << label << " ";
    for (unsigned i = 0; i < NUM_INPUTS; ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}


/* OLD TEST
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
*/
AP_GTEST_MAIN()