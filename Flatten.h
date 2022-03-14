#ifndef _FlattenH_
#define _FlattenH_

#include "Layer.h"
#include "Conv.h"

//using namespace std;

class Flatten : public Layer {
private:
	int size;
	int sl_1[3];
	vconvAL* input;
	std::vector<denseA>* upGradient;
	std::vector<std::vector<unsigned int>> ixs;
public:
	std::vector<denseA> activations;
	vconvAL gradient;
	Flatten(vconvAL* input);
	void forward();
	void backward();
	void initialiseParameters(std::string actF);
	void setUpGradient(std::vector<denseA>* upGradient);
	void endTraining();
};

#endif