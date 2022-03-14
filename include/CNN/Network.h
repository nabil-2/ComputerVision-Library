#ifndef _NetworkH_
#define _NetworkH_

#include "activations.h"
#include "Layer.h"
#include "Dense.h"
#include "Conv.h"
#include "Flatten.h"
#include "Softmax.h"
#include "Input.h"
#include "MultiDense.h"
#include "MultiDense2.h"
#include "MultiDenseInput.h"
#include "Pooling.h"
#include <string>
#include <vector>

class Network {
private:
	std::vector<std::vector<Layer*>> level;
	std::string actF = "ReLU";
	std::string costF = "cross-entropy";
	std::vector<std::vector<std::string>> encodeOutput(std::vector<denseA> output);
	convAL preprocess(convAL* img);
	float batchMean = 0;
	std::string avgE = "";
public:
	std::vector<std::vector<std::string>> synsetID;
	Network();
	~Network();
	int addLevel();
	int addLayer(std::string layer, int size, int inputLevel, int inputLayer);
	int addLayer(std::string layerType, int actMaps, int inputLevel, int inputLayer, int kernel, int stride, int padding);
	int addLayer(int kernel, std::string layerType, int inputLevel, int inputLayer);
	int addLayer(std::string layer, int inputLevel, int inputLayer);
	int addLayer(std::string layer, int *size);
	void intialiseParameters();
	void setActivationFunction(std::string actFunction);
	//void setInput(std::vector<convA> *input);
	void setInput(vconvAL* input);
	//void setInput(convA input);
	std::vector<std::vector<std::string>> predict();
	std::vector<denseA> predictVal();
	void epoch(std::vector<std::vector<std::string>> *data);
	void setCostFunction(std::string costFunction);
	std::vector<denseA> sortOutput(std::vector<denseA> output);
	void finishTraining();
	void exportNet(std::string filename);
	void importNet(std::string filename);
	void calcBatchMean(int batchSize, std::string imgPath);
};

#endif