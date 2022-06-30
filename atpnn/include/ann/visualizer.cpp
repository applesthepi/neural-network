#include "visualizer.hpp"

#include <SFML/Graphics.hpp>

// #define STB_IMAGE_IMPLEMENTATION
// #include <stb/stb_image.h>
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include <stb/stb_image_write.h>

constexpr int32_t neuron_diameter = 50;
constexpr int32_t neuron_space = 30;
constexpr int32_t network_padding = 50;
constexpr int32_t network_margin = 200;

void ann::visualize_nn(ann::nn_structure* structure, const std::string& file_name)
{
	uint16_t max_neuron_count = 0;

	for (uint8_t layer = 0; layer < structure->layer_count; layer++)
	{
		uint16_t neuron_count = structure->layer_neuron_count[layer];

		if (neuron_count > max_neuron_count)
			max_neuron_count = neuron_count;
	}

	sf::RenderTexture render_texture;
	render_texture.create(
		(network_padding * 2) + (network_margin * (structure->layer_count - 1)) + (neuron_diameter * structure->layer_count),
		(network_padding * 2) + (neuron_diameter * max_neuron_count) + (neuron_space * (max_neuron_count - 1)),
		sf::ContextSettings(0, 0, 8)
	);
	render_texture.clear(sf::Color(30, 30, 30, 255));

	float center = (float)render_texture.getSize().y * 0.5f;

	//
	// CONNECTIONS / WEIGHTS
	//

	sf::ConvexShape sp_connection;
	sp_connection.setPointCount(4);
	sp_connection.setFillColor(sf::Color::Blue);

	auto color_weight_positive = sf::Color(87, 102, 128, 255);
	auto color_weight_negitive = sf::Color(153, 43, 43, 255);

	for (uint8_t layer = 1; layer < structure->layer_count; layer++)
	{
		for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer]; layer_neuron++)
		{
			float neuron_count_p = structure->layer_neuron_count[layer];
			float length_p = (neuron_diameter * neuron_count_p) + (neuron_space * (neuron_count_p - 1));

			float neuron_count_c = structure->layer_neuron_count[layer - 1];
			float length_c = (neuron_diameter * neuron_count_c) + (neuron_space * (neuron_count_c - 1));

			for (uint16_t layer_neuron_con = 0; layer_neuron_con < structure->layer_neuron_count[layer - 1]; layer_neuron_con++)
			{
				float line_width = structure->layer_neuron_con_weight[layer][layer_neuron][layer_neuron_con] * 9.0f + 1.0f;

				if (line_width > 10.0f)
				{
					line_width = 10.0f;
					sp_connection.setFillColor(sf::Color(118, 123, 133, 255));
				}
				else if (line_width < -10.0f)
				{
					line_width = -10.0f;
					sp_connection.setFillColor(sf::Color(168, 76, 76, 255));
				}
				else if (line_width >= 0.0f)
				{
					float scaler = line_width * 0.1f;
					sp_connection.setFillColor(sf::Color(
						(uint8_t)((float)color_weight_positive.r * scaler),
						(uint8_t)((float)color_weight_positive.g * scaler),
						(uint8_t)((float)color_weight_positive.b * scaler),
						255
					));
				}
				else if (line_width < 0.0f)
				{
					float scaler = line_width * -0.1f;
					sp_connection.setFillColor(sf::Color(
						(uint8_t)((float)color_weight_negitive.r * scaler),
						(uint8_t)((float)color_weight_negitive.g * scaler),
						(uint8_t)((float)color_weight_negitive.b * scaler),
						255
					));
				}

				line_width = glm::abs(line_width);

				// PARENT POINTS (right)

				sp_connection.setPoint(3, sf::Vector2f(
					network_padding + (layer * (network_margin + neuron_diameter)) + (neuron_diameter * 0.5f),
					(layer_neuron * (neuron_diameter + neuron_space)) + center - (length_p * 0.5f) + (neuron_diameter * 0.5f)
				));

				sp_connection.setPoint(2, sf::Vector2f(
					network_padding + (layer * (network_margin + neuron_diameter)) + (neuron_diameter * 0.5f),
					(layer_neuron * (neuron_diameter + neuron_space)) + center - (length_p * 0.5f) + line_width + (neuron_diameter * 0.5f)
				));

				// CHILD POINTS (left)

				sp_connection.setPoint(0, sf::Vector2f(
					network_padding + ((layer - 1) * (network_margin + neuron_diameter)) + (neuron_diameter * 0.5f),
					(layer_neuron_con * (neuron_diameter + neuron_space)) + center - (length_c * 0.5f) + (neuron_diameter * 0.5f)
				));

				sp_connection.setPoint(1, sf::Vector2f(
					network_padding + ((layer - 1) * (network_margin + neuron_diameter)) + (neuron_diameter * 0.5f),
					(layer_neuron_con * (neuron_diameter + neuron_space)) + center - (length_c * 0.5f) + line_width + (neuron_diameter * 0.5f)
				));

				render_texture.draw(sp_connection);
			}
		}
	}

	//
	// NEURONS / ACTIVATIONS
	//

	sf::CircleShape sp_neuron;
	sp_neuron.setRadius(neuron_diameter / 2);
	sp_neuron.setFillColor(sf::Color::Black);
	sp_neuron.setOutlineColor(sf::Color::White);
	sp_neuron.setOutlineThickness(1.5f);

	for (uint8_t layer = 0; layer < structure->layer_count; layer++)
	{
		for (uint16_t layer_neuron = 0; layer_neuron < structure->layer_neuron_count[layer]; layer_neuron++)
		{
			float neuron_count = structure->layer_neuron_count[layer];
			float length = (neuron_diameter * neuron_count) + (neuron_space * (neuron_count - 1));

			sp_neuron.setPosition(sf::Vector2f(
				network_padding + (layer * (network_margin + neuron_diameter)),
				(layer_neuron * (neuron_diameter + neuron_space)) + center - (length * 0.5f)
			));

			// TODO: CHANGE TO BIAS NOT ACTIVATION
			sp_neuron.setFillColor(sf::Color(
				(uint8_t)(glm::clamp(structure->layer_neuron_activation[layer][layer_neuron], 0.0f, 1.0f) * 255.0f),
				(uint8_t)(glm::clamp(structure->layer_neuron_activation[layer][layer_neuron], 0.0f, 1.0f) * 255.0f),
				(uint8_t)(glm::clamp(structure->layer_neuron_activation[layer][layer_neuron], 0.0f, 1.0f) * 255.0f),
				255
			));

			render_texture.draw(sp_neuron);
		}
	}

	render_texture.display();
	auto image_capture = render_texture.getTexture().copyToImage();
	image_capture.saveToFile(file_name);
}