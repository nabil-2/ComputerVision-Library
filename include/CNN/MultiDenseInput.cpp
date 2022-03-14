#include "MultiDenseInput.h"
#include "MathCNN.h"
#include <random>

MultiDenseInput::MultiDenseInput(std::vector<denseA>* input_) : inputAct(input_) {
	size = (*inputAct)[0].size();
	sl_1 = size;
	activations = std::vector<denseA>(1, denseA(size));
	gradient = std::vector<denseA>(0);
	position.reserve(size);
	position = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
	upGradient = nullptr;
};
void MultiDenseInput::forward() {
	activations = *inputAct;
};
void MultiDenseInput::backward() {
	for (unsigned int i = 0; i < size; i++) { //this Layer
		for (unsigned int j = 0; j < dimensions; j++) { //dimensions
			float avgC = 0;
			for (unsigned int k = 0; k < activations.size(); k++) { //batches
				float sum = 0;
				for (unsigned int l = 0; l < posp1->size(); l++) { //next Layer
					float tmp = (*wGp1)[k][l][i];
					std::vector<float>* inPos = &(position[i]),
									  * outPos = &((*posp1)[l]);
					float distance = MathCNN::distance(inPos, outPos);
					if (!v2) {
						tmp *= (-1) * (*ap1)[l] / (2 * sqrt(distance + deltaM));
					}
					else {
						tmp *= (*ap1)[l];
					}
					tmp *= (position[i][j] - (*posp1)[l][j]) / distance;
					sum += tmp;
				}
				avgC += sum;
			}
			avgC /= activations.size();
			positionGradients[i][j] += avgC;
		}

	}
	gradient = *upGradient;
	updateParameters();
};
void MultiDenseInput::initialiseParameters(std::string actF) {
	if (v2) {
		float mean = (float)0;
		float dev = 1;
		if (actF == "ReLU") {
			dev = 1 / (1 * sqrt((float)sl_1)); // 1/alpha*sqrt(n)
		}
		std::default_random_engine generator;
		for (unsigned int i = 0; i < size; i++) {
			for (unsigned int j = 0; j < dimensions; j++) {
				position[i][j] = MathCNN::getRandomNormal(&mean, &dev, &generator);
			}
		}
	}
	positionGradients.reserve(size);
	positionVelocity1.reserve(size);
	positionVelocity2.reserve(size);
	positionGradients = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
	positionVelocity1 = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
	positionVelocity2 = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
};
void MultiDenseInput::setUpGradient(std::vector<denseA>* upGradient_) {
	upGradient = upGradient_;
};
void MultiDenseInput::updateParameters() {
	for (unsigned int i = 0; i < positionGradients.size(); i++) {
		for (unsigned int j = 0; j < positionGradients[i].size(); j++) {
			positionVelocity1[i][j] = beta_o1 * positionVelocity1[i][j] + ((1 - beta_o1) * positionGradients[i][j]);
			positionVelocity2[i][j] = beta_o2 * positionVelocity2[i][j] + ((1 - beta_o2) * pow(positionGradients[i][j], 2));
		}
	}
	for (unsigned int i = 0; i < position.size(); i++) {
		for (unsigned int j = 0; j < position[i].size(); j++) {
			float positionV1_h = positionVelocity1[i][j] / (1 - pow(beta_o1, epoch));
			float positionV2_h = positionVelocity2[i][j] / (1 - pow(beta_o2, epoch));
			position[i][j] -= eta * (positionV1_h / sqrt(positionV2_h + 0.00001));
		}
	}
};
void MultiDenseInput::endTraining() {
	positionGradients.clear();
	positionGradients.shrink_to_fit();
	positionVelocity1.clear();
	positionVelocity1.shrink_to_fit();
	positionVelocity2.clear();
	positionVelocity2.shrink_to_fit();
	gradient.clear();
	gradient.shrink_to_fit();
	activations.clear();
	activations.shrink_to_fit();
}