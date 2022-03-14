#ifndef _SoftmaxH_
#define _SoftmaxH_

#include "Layer.h"
#include "Dense.h"

//using namespace std;

class Softmax : public Layer {
private:
	int size, sl_1;
	std::vector<denseA>* input;
	std::vector<denseA>* upGradient;
public:
	std::vector<denseA> resultD;
	std::vector<denseA> activations;
	std::vector<denseA> gradient;
	Softmax(std::vector<denseA>* input);
	void forward();
	void backward();
	void initialiseParameters(std::string actF);
	void setUpGradient(std::vector<denseA>* upGradient);
};

#endif