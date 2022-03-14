#ifndef _MultiDense2H_
#define _MultiDense2H_

#include "Layer.h"
#include <string>

//using namespace std;

class MultiDense2 : public Layer {
private:
	int dimensions = 3;
	int size, sl_1;
	std::vector<denseA>* inputAct;
	std::vector<std::vector<float>>* inputPos;
	std::vector<denseA>* upGradient;

	float delta = 0.0001;
	std::vector<denseA> zws;
	std::vector<denseA> u;
	std::vector<denseA> zws_hat;
	denseA m, var;
	float deltaM = 0.1;
	float meanDistance = 1;

	denseA betaGradients;
	denseA betaVelocity1;
	denseA betaVelocity2;
	denseA gammaGradients;
	denseA gammaVelocity1;
	denseA gammaVelocity2;
	denseA betaMGradients;
	denseA betaMVelocity1;
	denseA betaMVelocity2;
	denseA alphaMGradients;
	denseA alphaMVelocity1;
	denseA alphaMVelocity2;
	std::vector<std::vector<float>> positionGradients;
	std::vector<std::vector<float>> positionVelocity1;
	std::vector<std::vector<float>> positionVelocity2;

	void updateParameters();
	float eta = 0.0001,
		beta_o1 = 0.9,
		beta_o2 = 0.999;
	int epoch = 1;

public:
	bool converged = false;
	denseA avgM, avgVar;
	denseA betaM;
	denseA alphaM;
	denseA gamma;
	denseA beta;
	std::vector<std::vector<float>> position;
	int ixLayer;
	bool lastLayer = false,
		nextW = true;
	float mean, * meanl_1;
	std::vector<denseA> activations;
	std::vector<denseA> gradient;
	std::vector<denseW> wG;
	std::vector<denseW>* wGp1;
	std::vector<float>* ap1;
	std::vector<std::vector<float>>* posp1;
	MultiDense2(int size, std::vector<denseA>* inputAct, std::vector<std::vector<float>>* inputPos, float* meanl_1, int ixLayer);
	void forward();
	void backward();
	void initialiseParameters(std::string actF);
	void setUpGradient(std::vector<denseA>* upGradient);
	void setInputPosition(std::vector<std::vector<float>>* inputPos);
	void endTraining();
	denseA getAvgMean();
	denseA getAvgVar();
};

#endif;