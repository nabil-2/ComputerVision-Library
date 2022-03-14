#include "Softmax.h"
#include "MathCNN.h"
#include <iostream>

Softmax::Softmax(std::vector<denseA> *input_) : input(input_) {
	size = (*input)[0].size();
	sl_1 = size;
	upGradient = nullptr;
	activations = std::vector<denseA>(1, denseA(size));
};
void Softmax::forward() {
	activations = std::vector<denseA>(input->size(), denseA(size));
	for (unsigned int i = 0; i < input->size(); i++) {
		MathCNN::softmax(&(*input)[i], &activations[i]);
	}
};
void Softmax::backward() {
	gradient = std::vector<denseA>(0);
	for (unsigned int i = 0; i < activations.size(); i++) {
		gradient.push_back(MathCNN::subtractVec(&activations[i], &resultD[i]));
	}
	/*vector<vector<float>> lGs;
	vector<int> ix_j;
	for (unsigned int i = 0; i < upGradient->size(); i++) {
		vector<float> *uG = &((*upGradient)[i]);
		vector<float> lG;
		float fzj = 0;
		for (unsigned int j = 0; j < uG->size(); j++) {
			if ((*uG)[j] != 0) {
				fzj = activations[i][j];
				ix_j.push_back(j);
			}
		}
		for (unsigned int j = 0; j < uG->size(); j++) {
			float fzi = activations[i][j];
			if ((*uG)[j] != 0) {
				fzi *= (1 - fzi);
			} else {
				fzi *= (-1) * fzj;
			}
			lG.push_back(fzi);
		}
		lGs.push_back(lG);
	}
	gradient = vector<denseA>(0);
	for (unsigned int i = 0; i < lGs.size(); i++) {
		gradient.push_back(denseA());
		for (unsigned int j = 0; j < lGs[i].size(); j++) {
			gradient[i].push_back(lGs[i][j] * (*upGradient)[i][ix_j[i]]);
		}
	}*/
};
void Softmax::initialiseParameters(std::string actF) {};
void Softmax::setUpGradient(std::vector<denseA>* upG) {
	upGradient = upG;
};