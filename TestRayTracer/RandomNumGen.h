#pragma once

#include <random>

class RandomNumGen
{
public:
	RandomNumGen()
		: gen(rd())
		, dis(0.0, 1.0)
	{
	}
	static double GetRand()
	{
		thread_local std::unique_ptr<RandomNumGen> randomNumGen;
		if(!randomNumGen)
			randomNumGen = std::unique_ptr<RandomNumGen>(new RandomNumGen());
		return randomNumGen->dis(randomNumGen->gen);
	}

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis;

};

class PreGenerated
{
public:
	PreGenerated(size_t size)
	{
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(0.0, 1.0);
		vec.resize(size);
		for (size_t i = 0; i < size; ++i)
			vec[i] = dis(gen);
	}
	const std::vector<float>& GetVector() const
	{
		return vec;
	}
private:
	std::vector<float> vec;
};

class RandAccessor
{
public:
	RandAccessor(size_t offset_, const float* arr_, size_t size_)
		: offset(offset_)
		, arr(arr_)
		, size(size_)
		, index(0)
	{}
	float Get() const
	{
		size_t i = (offset + index) % size;
		++index;
		return arr[i];
	}
private:
	size_t offset;
	const float* arr;
	size_t size;
	mutable size_t index;
};