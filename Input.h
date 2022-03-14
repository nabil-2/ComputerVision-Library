#ifndef _InputH_
#define _InputH_

#include "Layer.h"
#include "activations.h"

//using namespace std;

class Input : public Layer {
private:
	int *size;
public:
	vconvAL activations;
	Input(int *size);
	void forward();
	void backward();
	void initialiseParameters(std::string actF);
};

#endif