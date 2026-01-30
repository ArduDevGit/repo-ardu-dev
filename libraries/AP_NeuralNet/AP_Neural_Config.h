

#pragma once


static constexpr unsigned NUM_LAYERS = 3;

static constexpr unsigned NUM_INPUTS = 2;

// These need to stay in sync with NUM_LAYERS and the training data file topology i.e. "topology: 3 3 1"
// Plus 1 for a bias at every layer.
static constexpr unsigned NUM_NEURONS_INPUT_LAYER = NUM_INPUTS + 1;
static constexpr unsigned NUM_NEURONS_LAYER_ONE =   2 + 1;
// can add more layers here
//
static constexpr unsigned NUM_NEURONS_OUTPUT_LAYER = 1; //no bias in output layer

// Max number of neurons for each layer (up to 9 with 1 bias)
static constexpr unsigned MAX_NEURONS = 10;