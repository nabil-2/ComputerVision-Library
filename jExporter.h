#ifndef _jExporterH_
#define _jExporterH_

#include <vector>
#include <string>

namespace j {
	typedef std::vector<float> denseA;
	typedef std::vector<std::vector<std::vector<float>>> convA;
	typedef std::vector<std::vector<float>> denseW;
	typedef std::vector<std::vector<float>> featureMap;
}

struct denseExp {
	j::denseW* weights;
	j::denseA* beta, *gamma;
	j::denseA* avgM, *avgVar;
};

struct convExp {
	std::vector<j::convA>* kernel;
	std::vector<float>* beta, *gamma;
	std::vector<float>* biases;
	j::denseA* avgM, *avgVar;
};

struct inputExp {
	float* batchMean;
};

struct multiDenseExp {
	j::denseA* avgM, *avgVar;
	j::denseA* betaM, *alphaM;
	j::denseA* beta, *gamma;
	std::vector<std::vector<float>>* position;
};

struct multiDenseInputExp {
	std::vector<std::vector<float>>* position;
};

class jExporter {
public:
	static void exportData(std::string filename, denseExp layer);
	static void exportData(std::string filename, convExp layer);
	static void exportData(std::string filename, inputExp layer);
	static void exportData(std::string filename, multiDenseExp layer);
	static void exportData(std::string filename, multiDenseInputExp layer);

	static void importData(int i, std::string filename, j::denseW* weights, j::denseA* beta, j::denseA* gamma, j::denseA* avgM, j::denseA* avgVar); //dense
	static void importData(int i, std::string filename, std::vector<j::convA>* kernel, j::denseA* beta, j::denseA* gamma, j::denseA* biases, j::denseA* avgM, j::denseA* avgVar); //conv
	static void importData(int i, std::string filename, float* batchMean); //input
	static void importData(int i, std::string filename, j::denseA* avgM, j::denseA* avgVar, j::denseA* betaM, j::denseA* alphaM, j::denseA* beta, j::denseA* gamma, j::denseW* position); //multiDense
	static void importData(int i, std::string filename, j::denseW* position); //multiDenseInput

	//static void exportNet(std::string filename, float batchMean, std::vector< std::vector<Layer*>>* level);
	//static void importNet(std::string filename, float* batchMean, std::vector< std::vector<Layer*>>* level);
};

#endif