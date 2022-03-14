#ifndef _MultiDenseInputH_
#define _MultiDenseInputH_

#include "Layer.h"

//using namespace std;

class MultiDenseInput : public Layer {
private:
	int dimensions = 3;
	int size, sl_1;
	std::vector<denseA>* inputAct;
	std::vector<denseA>* upGradient;
	std::vector<std::vector<float>> positionGradients;
	std::vector<std::vector<float>> positionVelocity1;
	std::vector<std::vector<float>> positionVelocity2;
	float deltaM = 0.1;
	void updateParameters();
	float eta = 0.0001,
		beta_o1 = 0.9,
		beta_o2 = 0.999;
	int epoch = 1;
public:
	bool v2 = false;
	std::vector<denseA> activations;
	std::vector<denseA> gradient;
	std::vector<std::vector<float>> position;
	std::vector<denseW> wG;
	std::vector<denseW>* wGp1;
	std::vector<float>* ap1;
	std::vector<std::vector<float>>* posp1;
	MultiDenseInput(std::vector<denseA>* input);
	void forward();
	void backward();
	void initialiseParameters(std::string actF);
	void setUpGradient(std::vector<denseA>* upGradient);
	void endTraining();
};

#endif