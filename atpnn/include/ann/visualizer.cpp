#include "visualizer.hpp"
#include <SFML/Graphics.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

void ann::visualize_nn(ann::nn_structure* structure, const std::string& file_name)
{
#if 0
	uint8_t image[] = {
		0, 0, 200, 255,
		100, 100, 100, 255,
		100, 100, 100, 255,
		200, 0, 0, 255
	};

	stbi_write_png("test.png", 2, 2, 4, image, 0);
#endif

	sf::RenderWindow render_window;
	render_window.setSize(sf::Vector2u(100, 100));

	sf::RectangleShape sp;
	sp.setFillColor(sf::Color::Green);
	sp.setSize(sf::Vector2f(50.0f, 50.0f));

	render_window.draw(sp);

	render_window.display();
	auto image_capture = render_window.capture();
	image_capture.saveToFile("test.png");
}

void ann::draw_line(uint8_t* image, glm::vec<2, int32_t> p0, glm::vec<2, int32_t> p1, float thickness, glm::vec<4, uint8_t> color)
{

}