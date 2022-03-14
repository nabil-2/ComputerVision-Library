#ifndef _ConvH_
#define _ConvH_

#include "activations.h"
#include "MathCNN.h"
#include "Layer.h"
#include <thread>

//using namespace std;

class Conv : public Layer {
private:
	convA layerDim;
	//convAL layerDimx;
	int size[3];
	int sl_1[3];
	int stride, kernelS, padding;
	bool BN = true;
	vconvAL* input;
	vconvAL* upGradient;
	vconvAL zws;
	std::vector<float> m, var;
	int epoch = 1;
	float delta = 0.0001;
	float eta = 0.0001,
		  beta_o1 = 0.9,
		  beta_o2 = 0.999;
	vconvAL u, zws_hat;
	std::vector<convA> kernelGradient, kernelVel1, kernelVel2;
	std::vector<float> gammaGradient, gammaVel1, gammaVel2;
	std::vector<float> betaGradient, betaVel1, betaVel2;
	std::vector<float> biasesGradient, biasesVel1, biasesVel2;
	void updateParameters();
	//vector<thread> kernelThreads, gradientThreads;
public:
	bool converged;
	std::vector<convA> kernel;
	std::vector<float> gamma, beta;
	std::vector<float> biases;
	std::vector<float> avgM, avgVar;
	vconvAL activations;
	vconvAL gradient;
	Conv(int actMaps, vconvAL* input, int kernel, int stride, int padding);
	void forward();
	void backward();
	void initialiseParameters(std::string actF);
	void setUpGradient(vconvAL* upGradient);
	void disableBN();
	void endTraining();
	std::vector<float> getAvgMean();
	std::vector<float> getAvgVar();
};

#endif