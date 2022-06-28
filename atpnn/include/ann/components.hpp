#pragma once

#include <stdint.h>
#include <stdlib.h>

namespace ann
{
struct nn_structure
{
	uint8_t layer_count;
	uint16_t* layer_neuron_count;

	// Active signal that gets carried.
	float** layer_neuron_activation;

	// Bias for activation from prevoius neurons.
	float** layer_neuron_bias;

	// Previous connections to all the neurons before it.
	float*** layer_neuron_con_weight;
};

struct nn_mod
{
	float** layer_neuron_activation;
	float** layer_neuron_bias;
	float*** layer_neuron_con_weight;
};

struct nn_cmod
{
	float** layer_neuron_bias;
	float*** layer_neuron_con_weight;
};

void create_nn_structure(nn_structure* structure, uint8_t layer_count, uint16_t* layer_neuron_count);

void print_nn_neuron_info(uint8_t layer_count, uint16_t* layer_neuron_count, float** neuron_info);

void calc_neuron(nn_structure* structure, uint8_t layer, uint16_t neuron);

void calc_nn(nn_structure* structure, float* input, float* output);

void train_nn(nn_structure* structure, size_t iterations, size_t sample_count, size_t batch_count, float** sample_input, float** sample_output);

void train_sample(nn_structure* structure, nn_mod* mod, float* sample_input, float* sample_output);
}