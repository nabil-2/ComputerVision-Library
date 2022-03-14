#include "Pooling.h"

Pooling::Pooling(vconvAL* input_, int kernel_) : input(input_), kernel(kernel_) {
	sl_1[0] = (*input)[0].size();
	sl_1[1] = (*input)[0][0].size();
	sl_1[2] = (*input)[0][0][0].size();
	size[0] = MathCNN::convSize(sl_1[0], kernel, 0, kernel);
	size[1] = MathCNN::convSize(sl_1[1], kernel, 0, kernel);
	size[2] = sl_1[2];
	//std::vector<convA> acStd = std::vector<convA>(1, convA(size[0], std::vector<std::vector<float>>(size[1], std::vector<float>(size[2]))));		//convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2])));
	//activations = MathCNN::asStxxl(&acStd);
	activations.push_back(convA(size[0], std::vector<std::vector<float>>(size[1], std::vector<float>(size[2]))));
	//gradient = vector<convA>(0);		//convA(sl_1[0], vector<vector<float>>(sl_1[1], vector<float>(sl_1[2])));
	upGradient = nullptr;
};
void Pooling::forward() {
	//avgPooling
	activations.clear();
	for (unsigned int i = 0; i < input->size(); i++) {
		activations.push_back(convA(size[0], std::vector<std::vector<float>>(size[1], std::vector<float>(size[2]))));
	}
	for (unsigned int l = 0; l < input->size(); l++) { //batches
		for (unsigned int i = 0; i < size[2]; i++) { //depth
			for (unsigned int j = 0; j < size[0]; j++) { //width
				for (unsigned int k = 0; k < size[1]; k++) { //height
					float avg = 0;
					for (unsigned int k1 = 0; k1 < kernel; k1++) {
						for (unsigned int k2 = 0; k2 < kernel; k2++) {
							int J = k1 + j * kernel,
								K = k2 + k * kernel; //stride = kernel
							avg += (*input)[l][J][K][i];
						}
					}
					avg /= kernel * kernel;
					activations[l][j][k][i] = avg;
				}
			}
		}
	}
};
void Pooling::backward() {
	gradient.clear();
	for (unsigned int i = 0; i < input->size(); i++) {
		gradient.push_back(convA(sl_1[0], std::vector<std::vector<float>>(sl_1[1], std::vector<float>(sl_1[2]))));
	}
	for (unsigned int l = 0; l < input->size(); l++) { //batches
		for (unsigned int i = 0; i < size[2]; i++) { //depth(l-1 = l)
			for (unsigned int j = 0; j < size[0]; j++) { //width
				for (unsigned int k = 0; k < size[1]; k++) { //height
					for (unsigned int k1 = 0; k1 < kernel; k1++) {
						for (unsigned int k2 = 0; k2 < kernel; k2++) {
							int J = k1 + j * kernel,
								K = k2 + k * kernel; //stride = kernel
							gradient[l][J][K][i] = (*upGradient)[l][j][k][i] / (kernel * kernel);
						}
					}
				}
			}
		}
	}
};
void Pooling::initialiseParameters(std::string actF) {};
void Pooling::setUpGradient(vconvAL* upGradient_) {
	upGradient = upGradient_;
};