#include "Flatten.h"

Flatten::Flatten(vconvAL* input_) : input(input_) {
	sl_1[0] = (*input)[0].size();
	sl_1[1] = (*input)[0][0].size();
	sl_1[2] = (*input)[0][0][0].size();
	size = sl_1[0] * sl_1[1] * sl_1[2];
	activations = std::vector<denseA>(1, denseA(size));		//denseA(sX * sY * sZ);
	gradient = vconvAL();	//convA(sl_1[0], vector<vector<float>>(sl_1[1], vector<float>(sl_1[2])));
	upGradient = nullptr;
};
void Flatten::forward() {
	activations = std::vector<denseA>(0);
	activations.reserve(input->size());
	for (unsigned int i = 0; i < input->size(); i++) {
		activations.push_back(denseA(0));
		for (unsigned int j = 0; j < (*input)[i].size(); j++) {
			for (unsigned int k = 0; k < (*input)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*input)[i][j][k].size(); l++) {
					activations[i].push_back((*input)[i][j][k][l]);
				}
			}
		}
	}
};
void Flatten::backward() {
	gradient = *input;
	for (unsigned int i = 0; i < upGradient->size(); i++) {
		for (unsigned int j = 0; j < (*upGradient)[i].size(); j++) {
			int I = ixs[j][0],
				J = ixs[j][1],
				K = ixs[j][2];
			gradient[i][I][J][K] = (*upGradient)[i][j];
		}
	}
};
void Flatten::initialiseParameters(std::string actF) {
	for (unsigned int j = 0; j < sl_1[0]; j++) {
		for (unsigned int k = 0; k < sl_1[1]; k++) {
			for (unsigned int l = 0; l < sl_1[2]; l++) {
				ixs.push_back(std::vector<unsigned int>({ j, k, l }));
			}
		}
	}
};
void Flatten::setUpGradient(std::vector<denseA>* upGradient_) {
	upGradient = upGradient_;
};
void Flatten::endTraining() {
	activations.clear();
	activations.shrink_to_fit();
};