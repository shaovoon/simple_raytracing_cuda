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
