#include <ann/ann.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>

constexpr size_t td_count = 100000;
constexpr size_t tds_count = 10;

int main()
{
	// GENERATE TRAINING DATA

	static std::random_device device_rng;
	static std::mt19937 device(device_rng());
	static std::uniform_int_distribution<int> dis_04(0, 1);

	float** sample_inputs = (float**)malloc(sizeof(float*) * td_count);
	float** sample_outputs = (float**)malloc(sizeof(float*) * td_count);

	for (size_t i = 0; i < td_count; i++)
	{
		sample_inputs[i] = (float*)malloc(sizeof(float) * 5);
		sample_outputs[i] = (float*)malloc(sizeof(float) * 5);

		memset(sample_inputs[i], 0, sizeof(float) * 5);
		memset(sample_outputs[i], 0, sizeof(float) * 5);

		// int rng = dis_04(device);
		int rng = 0;

		sample_inputs[i][rng] = 1.0f;
		sample_outputs[i][rng] = 1.0f;
	}

	float test_out[5] = { 0.0f };

	// CREATE NN

	ann::nn_structure structure;
	ann::create_nn_structure(&structure, 4, new uint16_t[4]{
		5, 5, 5, 5
	});

	std::cout << "TEST SAMPLE CASE\n"
		<< sample_inputs[0][0] << " "
		<< sample_inputs[0][1] << " "
		<< sample_inputs[0][2] << " "
		<< sample_inputs[0][3] << " "
		<< sample_inputs[0][4] << "\n\n" << std::flush;

	ann::calc_nn(&structure, sample_inputs[0], test_out);
	std::cout << "TEST\n";
	for (uint8_t i = 0; i < 5; i++)
		std::cout << test_out[i] << " ";
	std::cout << "\n\n" << std::flush;

	ann::train_nn(&structure, 1, td_count, tds_count, sample_inputs, sample_outputs);

	ann::calc_nn(&structure, sample_inputs[0], test_out);
	std::cout << "TEST\n";
	for (uint8_t i = 0; i < 5; i++)
		std::cout << test_out[i] << " ";
	std::cout << "\n\n" << std::flush;

	ann::visualize_nn(&structure, "test.png");

	return 0;
}