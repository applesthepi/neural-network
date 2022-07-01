#include <ann/ann.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>

constexpr size_t td_count = 100000;
constexpr size_t tds_count = 1;
constexpr uint16_t input_layer = 3;

int main()
{
	// GENERATE TRAINING DATA

	static std::random_device device_rng;
	static std::mt19937 device(device_rng());
	static std::uniform_int_distribution<int> dis_04(0, 2);

	float** sample_inputs = (float**)malloc(sizeof(float*) * td_count);
	float** sample_outputs = (float**)malloc(sizeof(float*) * td_count);

	for (size_t i = 0; i < td_count; i++)
	{
		sample_inputs[i] = (float*)malloc(sizeof(float) * input_layer);
		sample_outputs[i] = (float*)malloc(sizeof(float) * input_layer);

		memset(sample_inputs[i], 0, sizeof(float) * input_layer);
		memset(sample_outputs[i], 0, sizeof(float) * input_layer);

		int rng = dis_04(device);
		// int rng = 0;

		sample_inputs[i][rng] = 1.0f;
		sample_outputs[i][rng] = 1.0f;
	}

	float test_out[input_layer] = { 0.0f };

	// CREATE NN

	ann::nn_structure structure;
	ann::create_nn_structure(&structure, 5, new uint16_t[5]{
		input_layer, 9, 5, 5, 3
	});

	// std::cout << "TEST SAMPLE CASE\n"
	// 	<< sample_inputs[0][0] << " "
	// 	<< sample_inputs[0][1] << " "
	// 	<< sample_inputs[0][2] << " "
	// 	<< sample_inputs[0][3] << " "
	// 	<< sample_inputs[0][4] << "\n\n" << std::flush;

	ann::calc_nn(&structure, sample_inputs[0], test_out);
	std::cout << "TEST\n";
	for (uint8_t i = 0; i < input_layer; i++)
		std::cout << test_out[i] << " ";
	std::cout << "\n\n" << std::flush;

	ann::train_nn(&structure, 1000, td_count, tds_count, 100, sample_inputs, sample_outputs);

	ann::calc_nn(&structure, sample_inputs[0], test_out);
	std::cout << "TEST\n";
	for (uint8_t i = 0; i < input_layer; i++)
		std::cout << test_out[i] << " ";
	std::cout << "\n\n" << std::flush;

	for (size_t i = 0; i < 10; i++)
	{
		ann::calc_nn(&structure, sample_inputs[i], test_out);
		ann::visualize_nn(&structure, "result_" + std::to_string(i) + ".png");
	}

	return 0;
}