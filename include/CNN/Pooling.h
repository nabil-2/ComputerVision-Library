#ifndef _PoolingH_
#define _PoolingH_

#include "Layer.h"
#include "activations.h"
#include "MathCNN.h"

//using namespace std;

class Pooling : public Layer {
private:
	int size[3];
	int sl_1[3];
	int kernel;
	vconvAL* input;
	vconvAL* upGradient;
public:
	vconvAL activations;
	vconvAL gradient;
	Pooling(vconvAL* input, int kernel);
	void forward();
	void backward();
	void initialiseParameters(std::string actF);
	void setUpGradient(vconvAL* upGradient);
};

#endif