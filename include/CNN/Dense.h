#ifndef _DenseH_
#define _DenseH_

#include "Layer.h"
#include <string>
#include <thread>

//using namespace std;

class Dense : public Layer {
private:
	int size, sl_1;
	std::vector<denseA>* input;
	std::vector<denseA>* upGradient;
	
	std::vector<std::vector<float>> zws;
	std::vector<denseA> u;
	std::vector<denseA> zws_hat;
	denseA m, var;
	float delta = 0.0001;

	denseA betaGradients;
	denseA betaVelocity1;
	denseA betaVelocity2;
	denseA gammaGradients;
	denseA gammaVelocity1;
	denseA gammaVelocity2;
	denseW weightGradients;
	denseW weightVelocity1;
	denseW weightVelocity2;

	void updateParameters();
	float eta = 0.0001,
		  beta_o1 = 0.9,
		  beta_o2 = 0.999;
	int epoch = 1;	
public:
	bool converged = false;
	denseA beta;
	denseA gamma;
	denseW weights;
	denseA avgM,
		   avgVar;
	bool lastLayer = false;
	std::vector<denseA> activations;
	std::vector<denseA> gradient;
	Dense(int size, std::vector<denseA>* input);
	void forward() ;
	void backward();
	void initialiseParameters(std::string actF);
	void setUpGradient(std::vector<denseA>* upGradient);
	void endTraining();
	denseA getAvgMean();
	denseA getAvgVar();
};

#endif;