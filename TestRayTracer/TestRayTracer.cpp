//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#include <iostream>
#include "sphere.h"
#include <cfloat>
#include "camera.h"
#include "material.h"
#include "RandomNumGen.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>
#include <execution>
#include <memory>

class timer
{
public:
	timer() = default;
	void start(const std::string& text_)
	{
		text = text_;
		begin = std::chrono::high_resolution_clock::now();
	}
	void stop()
	{
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << std::setw(20) << text << " timing:" << std::setw(5) << ms << "ms" << std::endl;
	}

private:
	std::string text;
	std::chrono::high_resolution_clock::time_point begin;
};

bool world_hit(std::vector <sphere>& world, const ray& r, float tmin, float tmax, hit_record& rec){
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = tmax;
	for (int i = 0; i < world.size(); i++) {
		if (world[i].hit(r, tmin, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

vec3 color(const ray& r, std::vector <sphere>& world, int depth, const RandAccessor& rand) {
	hit_record rec;
	if (world_hit(world, r, 0.001, FLT_MAX, rec)) {
		ray scattered;
		vec3 attenuation;
		if (depth < 50 && rec.mat->scatter(r, rec, attenuation, scattered, rand)) {
			return attenuation * color(scattered, world, depth + 1, rand);
		}
		else {
			return vec3(0, 0, 0);
		}
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}


std::vector<sphere>  random_scene() {
	std::vector<sphere> list;
	list.push_back(sphere(vec3(0, -1000, 0), 1000, material(material_type::lambertian,  vec3(0.5, 0.5, 0.5), 0.0f, 0.0f)));
	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = RandomNumGen::GetRand();
			vec3 center(a + 0.9 * RandomNumGen::GetRand(), 0.2, b + 0.9 * RandomNumGen::GetRand());
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {  // diffuse
					list.push_back(sphere(center, 0.2, material(material_type::lambertian, vec3(RandomNumGen::GetRand() * RandomNumGen::GetRand(), RandomNumGen::GetRand() * RandomNumGen::GetRand(), RandomNumGen::GetRand() * RandomNumGen::GetRand()), 0.0f, 0.0f)));
				}
				else if (choose_mat < 0.95) { // metal
					list.push_back(sphere(center, 0.2,
						material(material_type::metal, vec3(0.5 * (1 + RandomNumGen::GetRand()), 0.5 * (1 + RandomNumGen::GetRand()), 0.5 * (1 + RandomNumGen::GetRand())), 0.5 * RandomNumGen::GetRand(), 0.0f)));
				}
				else {  // glass
					list.push_back(sphere(center, 0.2, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
				}
			}
		}
	}

	list.push_back(sphere(vec3(0, 1, 0), 1.0, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
	list.push_back(sphere(vec3(-4, 1, 0), 1.0, material(material_type::lambertian, vec3(0.4, 0.2, 0.1), 0.0f, 0.0f)));
	list.push_back(sphere(vec3(4, 1, 0), 1.0, material(material_type::metal, vec3(0.7, 0.6, 0.5), 0.0, 0.0)));

	return list;
}

int main() {
	int nx = 256;
	int ny = 256;
	int ns = 50;

	timer stopwatch;
	stopwatch.start("ray_tracer_init");

	std::vector<sphere> world;
	float R = cos(M_PI / 4);
	world.push_back(sphere(vec3(0, 0, -1), 0.5, material(material_type::lambertian, vec3(0.1, 0.2, 0.5), 0.0f, 0.0f)));
	world.push_back(sphere(vec3(0, -100.5, -1), 100, material(material_type::lambertian, vec3(0.8, 0.8, 0.0), 0.0f, 0.0f)));
	world.push_back(sphere(vec3(1, 0, -1), 0.5, material(material_type::metal, vec3(0.8, 0.6, 0.2), 0.0, 0.0f)));
	world.push_back(sphere(vec3(-1, 0, -1), 0.5, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
	world.push_back(sphere(vec3(-1, 0, -1), -0.45, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
	world = random_scene();

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.1;

	camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus);

	std::vector<unsigned int> pixelsSrc;
	pixelsSrc.resize(nx * ny);
	int stride = nx;
	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			int index = ((ny - 1) - j) * stride + i;
		
			pixelsSrc[index] = ((i & 0xffff) << 16) | (j & 0xffff);
		}
	}

	PreGenerated preGenerated(5000);
	const auto& vec = preGenerated.GetVector();
	const float* arr = vec.data();
	size_t rand_size = vec.size();

	stopwatch.stop();

	stopwatch.start("ray_tracer");

	std::for_each(std::execution::par, pixelsSrc.begin(), pixelsSrc.end(), [&](unsigned int& pixel) {
		int j = pixel & 0xffff;
		int i = (pixel & 0xffff0000) >> 16;

		RandAccessor rand(j+i, arr, rand_size);

		vec3 col(0, 0, 0);
		for (int s = 0; s < ns; s++) {
			float u = float(i + rand.Get()) / float(nx);
			float v = float(j + rand.Get()) / float(ny);
			ray r = cam.get_ray(u, v, rand);
			col += color(r, world, 0, rand);
		}
		col /= float(ns);
		col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
		int ir = int(255.99 * col[0]);
		int ig = int(255.99 * col[1]);
		int ib = int(255.99 * col[2]);

		int index = ((ny - 1) - j) * stride + i;
		pixel = (0xff000000 | (ib << 16) | (ig << 8) | ir);

	});

	stopwatch.stop();

	int channels = 4;
	stbi_write_png("c:\\temp\\ray_trace.png", nx, ny, channels, pixelsSrc.data(), nx * channels);

	return 0;
}

