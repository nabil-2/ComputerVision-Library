#ifndef _LayerH_
#define _LayerH_

#include "activations.h"
#include <string>
#include <vector>

//using namespace std;

class Layer {
public:
	void BatchNorm(std::vector<std::vector<float>>* z, denseA* mean, denseA* stddev, float delta);
	void BatchNorm(std::vector<std::vector<float>>* z, denseA mean, denseA var, float delta);
	void BatchNorm(vconvAL* z, std::vector<float>* mean, std::vector<float>* stddev, float delta);
	void BatchNorm(vconvAL* z, std::vector<float> mean, std::vector<float> var, float delta);
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void initialiseParameters(std::string actF) = 0;
};

#endif