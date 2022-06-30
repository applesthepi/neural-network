#include "components.hpp"

#include "visualizer.hpp"

#include <sstream>
#include <cstring>
#include <string>
#include <iomanip>
#include <random>
#include <iostream>
#include <glm/glm.hpp>

void ann::create_nn_structure(nn_structure* structure, uint8_t layer_count, uint16_t* layer_neuron_count)
{
	static std::random_device device_rng;
	static std::mt19937 device(device_rng());
	static std::uniform_real_distribution<float> dis_01(0.0f, 1.0f);
	static std::uniform_real_distribution<float> dis_11(-1.0f, 1.0f);

	structure->layer_count = layer_count;
	structure->layer_neuron_count = layer_neuron_count;

	structure->layer_neuron_activation = (float**)malloc(sizeof(float*) * layer_count);
	structure->layer_neuron_bias = (float**)malloc(sizeof(float*) * layer_count);
	structure->layer_neuron_con_weight = (float***)malloc(sizeof(float**) * layer_count);

	for (uint8_t layer = 0; layer < layer_count; layer++)
	{
		structure->layer_neuron_activation[layer] = (float*)malloc(sizeof(float) * layer_count);
		structure->layer_neuron_bias[layer] = (float*)malloc(sizeof(float) * layer_count);
		structure->layer_neuron_con_weight[layer] = (float**)malloc(sizeof(float*) * layer_count);

		memset(structure->layer_neuron_activation[layer], 0, sizeof(float) * layer_count);

		for (uint16_t layer_neuron = 0; layer_neuron < layer_neuron_count[layer]; layer_neuron++)
		{
			// BIAS
			
			structure->layer_neuron_bias[layer][layer_neuron] = 0.0f;//dis_11(device);

			// WEIGHTS

			if (layer == 0)
				structure->layer_neuron_con_weight[layer][layer_neuron] = nullptr;
			else
			{
				size_t layer_neuron_con_count = layer_neuron_count[layer - 1];
				structure->layer_neuron_con_weight[layer][layer_neuron] = (float*)malloc(sizeof(float) * layer_neuron_con_count);

				for (uint16_t layer_neuron_con = 0; layer_neuron_con < layer_neuron_con_count; layer_neuron_con++)
					structure->layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con] = dis_11(device);
			}
		}
	}
}

void ann::print_nn_neuron_info(uint8_t layer_count, uint16_t* layer_neuron_count, float** neuron_info)
{
	size_t max_neuron_count = 0;

	for (uint8_t i = 0; i < layer_count; i++)
	{
		if (layer_neuron_count[i] > max_neuron_count)
			max_neuron_count = layer_neuron_count[i];
	}

	std::vector<std::string> lines;
	lines.resize(max_neuron_count);

	for (uint16_t i = 0; i < max_neuron_count; i++)
	{
		for (uint8_t a = 0; a < layer_count; a++)
		{
			if (layer_neuron_count[a] >= i + 1)
			{
				std::ostringstream stream;
				stream << " " << std::fixed << std::setprecision(3) << neuron_info[a][i] << " ";
				lines[i] += stream.str();
			}
			else
			{
				lines[i] += "       ";
			}
		}
	}

	for (auto& line : lines)
		std::cout << line << "\n";
	
	std::cout << "\n" << std::flush;
}

void ann::calc_neuron(nn_structure* structure, uint8_t layer, uint16_t neuron)
{
	float weighted_sum = 0.0f;

	for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer - 1]; layer_neuron++)
	{
		weighted_sum +=
			structure->layer_neuron_activation[layer - 1][layer_neuron] *
			structure->layer_neuron_con_weight[layer][neuron][layer_neuron];
	}

	weighted_sum += structure->layer_neuron_bias[layer][neuron];
	structure->layer_neuron_activation[layer][neuron] = weighted_sum / (1.0f + std::abs(weighted_sum)) * 0.5f + 0.5f;
}

void ann::calc_nn(nn_structure* structure, float* input, float* output)
{
	memcpy(structure->layer_neuron_activation[0], input, sizeof(float) * structure->layer_neuron_count[0]);
	
	for (uint8_t layer = 1; layer < structure->layer_count; layer++)
		for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer]; layer_neuron++)
			calc_neuron(structure, layer, layer_neuron);
	
	memcpy(output, structure->layer_neuron_activation[structure->layer_count - 1], sizeof(float) * structure->layer_neuron_count[structure->layer_count - 1]);
}

void ann::train_nn(nn_structure* structure, size_t iterations, size_t sample_count, size_t batch_count, size_t visual_mod, float** sample_input, float** sample_output)
{
	ann::nn_mod* mod = (ann::nn_mod*)malloc(sizeof(ann::nn_mod) * (batch_count + 1));

	for (size_t i = 0; i < batch_count + 1; i++)
	{
		mod[i].layer_neuron_activation = (float**)malloc(sizeof(float*) * structure->layer_count);
		mod[i].layer_neuron_bias = (float**)malloc(sizeof(float*) * structure->layer_count);
		mod[i].layer_neuron_con_weight = (float***)malloc(sizeof(float**) * structure->layer_count);

		for (uint8_t layer = 0; layer < structure->layer_count; layer++)
		{
			mod[i].layer_neuron_activation[layer] = (float*)malloc(sizeof(float) * structure->layer_neuron_count[layer]);
			mod[i].layer_neuron_bias[layer] = (float*)malloc(sizeof(float) * structure->layer_neuron_count[layer]);
			mod[i].layer_neuron_con_weight[layer] = (float**)malloc(sizeof(float*) * structure->layer_neuron_count[layer]);

			memset(mod[i].layer_neuron_activation[layer], 0, sizeof(float) * structure->layer_neuron_count[layer]);
			memset(mod[i].layer_neuron_bias[layer], 0, sizeof(float) * structure->layer_neuron_count[layer]);

			for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer]; layer_neuron++)
			{
				if (layer == 0)
					mod[i].layer_neuron_con_weight[layer][layer_neuron] = nullptr;
				else
				{
					size_t layer_neuron_con_count = structure->layer_neuron_count[layer - 1];
					
					mod[i].layer_neuron_con_weight[layer][layer_neuron] = (float*)malloc(sizeof(float) * layer_neuron_con_count);
					memset(mod[i].layer_neuron_con_weight[layer][layer_neuron], 0, sizeof(float) * layer_neuron_con_count);
				}
			}
		}
	}

	size_t sample_idx = 0;

	for (size_t i = 0; i < iterations; i++)
	{
		// TRAIN SAMPLES
		
		for (size_t a = 0; a < batch_count; a++)
		{
			ann::train_sample(structure, mod + a,
				sample_input[sample_idx],
				sample_output[sample_idx]
			);

			sample_idx++;
			
			if (sample_idx >= sample_count)
				sample_idx = 0;
		}

		// COLLECT SAMPLE CHANGES

		for (size_t a = 0; a < batch_count; a++)
		{
			for (uint8_t layer = 0; layer < structure->layer_count; layer++)
			{
				for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer]; layer_neuron++)
				{
					mod[batch_count].layer_neuron_bias[layer][layer_neuron] += mod[a].layer_neuron_bias[layer][layer_neuron];

					if (layer > 0)
					{
						for (uint16_t layer_neuron_con = 0; layer_neuron_con < structure->layer_neuron_count[layer - 1]; layer_neuron_con++)
						{
							mod[batch_count].layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con] +=
								mod[a].layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con];
						}
					}

					memset(mod[a].layer_neuron_con_weight[layer][layer_neuron], 0, sizeof(float) * structure->layer_neuron_count[layer - 1]);
				}

				memset(mod[a].layer_neuron_activation[layer], 0, sizeof(float) * structure->layer_neuron_count[layer]);
				memset(mod[a].layer_neuron_bias[layer], 0, sizeof(float) * structure->layer_neuron_count[layer]);
			}
		}

		// APPLY SAMPLE CHANGES

		for (uint8_t layer = 0; layer < structure->layer_count; layer++)
		{
			for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer]; layer_neuron++)
			{
				// float& s_bias = structure->layer_neuron_bias[layer][layer_neuron];
				// s_bias += mod[batch_count].layer_neuron_bias[layer][layer_neuron] / batch_count;

				if (layer > 0)
				{
					for (uint16_t layer_neuron_con = 0; layer_neuron_con < structure->layer_neuron_count[layer - 1]; layer_neuron_con++)
					{
						float& s_con = structure->layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con];
						float& m_con = mod[batch_count].layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con];
						// float dcon = m_con * (glm::clamp(-1.0f * glm::abs(s_con) + 1.0f, 0.0f, 0.9f) + 0.1f);
						float dcon = m_con;
						// std::cout << m_con << "\n";
						s_con += dcon;
						s_con *= 0.99f;
						if (s_con > 1.0f)
							s_con = 1.0f;
						else if (s_con < -1.0f)
							s_con = -1.0f;
					}
				}
				
				memset(mod[batch_count].layer_neuron_con_weight[layer][layer_neuron], 0, sizeof(float) * structure->layer_neuron_count[layer - 1]);
			}

			memset(mod[batch_count].layer_neuron_bias[layer], 0, sizeof(float) * structure->layer_neuron_count[layer]);
		}

		if (visual_mod > 0 && i % visual_mod == 0)
			ann::visualize_nn(structure, "nn" + std::to_string(i) + ".png");
	}
}

void ann::train_sample(nn_structure* structure, nn_mod* mod, float* sample_input, float* sample_output)
{
	float* output = (float*)alloca(sizeof(float) * structure->layer_neuron_count[structure->layer_count - 1]);
	calc_nn(structure, sample_input, output);

	for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[structure->layer_count - 1]; layer_neuron++)
	{
		// Error MUST be 0-1
		float error = sample_output[layer_neuron] - structure->layer_neuron_activation[structure->layer_count - 1][layer_neuron];
		// error = std::pow(error, 2.0f);
		mod->layer_neuron_activation[structure->layer_count - 1][layer_neuron] = error;
		// std::cout << error << "\n";
	}

	for (uint8_t layer = structure->layer_count - 1; layer > 0; layer--)
	{
		for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer]; layer_neuron++)
		{
			float error = mod->layer_neuron_activation[layer][layer_neuron];// / (float)structure->layer_neuron_count[layer];
			error = glm::clamp(glm::pow(error, 3.0f), -1.0f, 1.0f);
			//float error = std::pow(mod->layer_neuron_activation[layer][layer_neuron] / (float)structure->layer_neuron_count[layer], 2.0f);

			// std::cout << error << "\n";
			// mod->layer_neuron_bias[layer][layer_neuron] += cost * 0.1f;

			for (uint16_t layer_neuron_con = 0; layer_neuron_con < structure->layer_neuron_count[layer - 1]; layer_neuron_con++)
			{
				float& s_act = structure->layer_neuron_activation[layer - 1][layer_neuron_con];
				float& s_con = structure->layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con];
				float& m_act = mod->layer_neuron_activation[layer - 1][layer_neuron_con];
				float& m_con = mod->layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con];
				// float con_dir = (std::round(s_con * 0.5f + 0.5f) * 2.0f - 1.0f);
				float con_dir = 1.0f;

				m_con += error * (s_act - 0.5f) * 1.0f;
				m_act += ((s_act - 0.5f) + (error * 0.1f)) * 0.5f;
				// m_act += s_con;
				// std::cout << m_act << "\n";
			}
		}
	}

	// for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[0]; layer_neuron++)
	// 	mod->layer_neuron_bias[0][layer_neuron] += mod->layer_neuron_activation[0][layer_neuron] * 0.5f;
}