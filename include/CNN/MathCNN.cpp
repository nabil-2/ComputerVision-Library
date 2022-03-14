#include "MathCNN.h"
#include <random>
#include <vector>
#include <algorithm>
#include <cctype>
#include <string>
#include <regex>
#include <time.h>;

//using namespace std;

MathCNN::MathCNN() {};
std::vector<float> MathCNN::MVProduct(const std::vector<std::vector<float>>* matrix, const std::vector<float>* vec) { //matrix: row<column>
	std::vector<float> result;
	for (unsigned int i = 0; i < (*matrix).size(); i++) {
		float sum = 0;
		int j = 0;
		for (auto it2 = vec->begin(); it2 != vec->end(); it2++) {
			sum += ((*matrix)[i][j]) * (*it2);
			j++;
		}
		result.push_back(sum);
	}
	return result;
};
std::vector<float> MathCNN::addVec(std::vector<float>* vec1, std::vector<float>* vec2) {
	std::vector<float> result;
	if (vec1->size() != vec2->size()) return result;
	for (unsigned int i = 0; i < vec1->size(); i++) {
		result.push_back((*vec1)[i] + (*vec2)[i]);
	}
	return result;
};
float MathCNN::getRandomNormal(const float* mean, const float* dev, std::default_random_engine* generator) {
	//default_random_engine generator;
	std::normal_distribution<float> distribution(*mean, *dev);
	return distribution(*generator);
}
float MathCNN::ReLU(float z) {
	if (z > 0) return z;
	return 0;
}
void MathCNN::ReLU(float* z) {
	if (*z < 0) *z = 0;
}
void MathCNN::ReLU(std::vector<float>* z) {
	for (unsigned int i = 0; i < z->size(); i++) {
		if ((*z)[i] < 0) (*z)[i] = 0;
	}
}
void MathCNN::ReLU_(std::vector<float>* z) {
	for (unsigned int i = 0; i < z->size(); i++) {
		if ((*z)[i] < 0) {
			(*z)[i] = 0;
		}
		else {
			(*z)[i] = 1;
		}
	}
}
void MathCNN::softmax(const std::vector<float>* in, std::vector<float>* out) {
	float inputSum = 0;
	for (unsigned int i = 0; i < in->size(); i++) {
		inputSum += exp((*in)[i]);
	}
	for (unsigned int i = 0; i < out->size(); i++) {
		(*out)[i] = exp((*in)[i]) / inputSum;
	}
};

void MathCNN::dotProduct(const std::vector<float>* in1, const std::vector<float>* in2, std::vector<float>* result) {
	for (unsigned int i = 0; i < in1->size(); i++) {
		(*result)[i] = (*in1)[i] * (*in2)[i];
	}
};
std::vector<std::vector<float>> MathCNN::transposeM(const std::vector<std::vector<float>>* matrix) {
	std::vector<std::vector<float>> transposed((*(matrix))[0].size(), std::vector<float>(matrix->size()));
	for (unsigned int i = 0; i < matrix->size(); i++) {
		for (unsigned int j = 0; j < (*(matrix))[i].size(); j++) {
			transposed[j][i] = (*(matrix))[i][j];
		}
	}
	return transposed;
};
int MathCNN::position(std::string letter) {
	std::vector<int> charIx(alphabet.size());
	for (unsigned int i = 0; i < charIx.size(); i++) {
		charIx[i] = i + 1;
	}
	auto it = find(alphabet.begin(), alphabet.end(), letter);
	return it - alphabet.begin();
}
std::string MathCNN::encode(std::string input) {
	reverse(input.begin(), input.end());
	std::vector<std::string> split;
	std::string exception = "";
	if (input.size() % 2 != 0) {
		std::string exception = "x";
		input += "0";
	}
	for (unsigned i = 0; i < input.size(); i+=2) {
		std::string tmp = std::string(1, input[i]) + std::string(1, input[i+1]);
		split.push_back(tmp);
	}
	std::string result = "";
	for (unsigned int i = 0; i < split.size(); i++) {
		int number = stoi(split[i]);
		int letterIx = number % 26;
		std::string digit = std::to_string((int) stoi(split[i])/26);
		result += alphabet[letterIx] + digit;
	}
	result += exception;
	return result;
}

std::string MathCNN::decode(std::string input) {
	std::vector<std::string> split;
	std::string result = "";
	int size = input.size();
	bool exception = false;
	if (std::string(1, input[size - 1]) == "x") {
		input.erase(size - 1, 1);
		exception = true;
	}
	for (unsigned i = 0; i < input.size(); i += 2) {
		std::string tmp = std::string(1, input[i]) + std::string(1, input[i + 1]);
		split.push_back(tmp);
	}
	for (unsigned int i = 0; i < split.size(); i++) {
		std::string letter = std::string(1, split[i][0]),
			   digit = std::string(1, split[i][1]);
		int number = position(letter) + stoi(digit)*26;
		std::string tmp = std::to_string(number);
		if (tmp.size() == 1) {
			tmp = "0" + tmp;
		}
		result += tmp;
	}
	if(exception) result.erase(result.size() - 1, 1);
	reverse(result.begin(), result.end());
	return result;
}


float MathCNN::scalarProduct(const std::vector<float>* vec1, const std::vector<float>* vec2) {
	float sum = 0;
	for (unsigned int i = 0; i < vec1->size(); i++) {
		sum += (*vec1)[i] * (*vec2)[i];
	}
	return sum;
};

std::vector<std::vector<float>> MathCNN::MMProduct(const std::vector<std::vector<float>>* matrix1, const std::vector<std::vector<float>>* matrix2) {
	std::vector<std::vector<float>> result;
	for (unsigned int i = 0; i < matrix1->size(); i++) {
		result.push_back(std::vector<float>());
		//vec1 = matrix1[i]
		for (unsigned int j = 0; j < (*matrix2)[0].size(); j++) {
			std::vector<float> vec2;
			for (unsigned k = 0; k < matrix2->size(); k++) {
				vec2.push_back((*matrix2)[k][j]);
			}
			result[i].push_back(scalarProduct(&((*matrix1)[i]), &vec2));
		}
	}
	return result;
};


void MathCNN::MVaddColumn(std::vector<std::vector<float>>* matrix, const std::vector<float>* vec) {
	for (unsigned int i = 0; i < matrix->size(); i++) {
		for (unsigned int j = 0; j < (*matrix)[i].size(); j++) {
			(*matrix)[i][j] += (*vec)[i];
		}
	}
};

void MathCNN::MVmultiplyColumn(std::vector<std::vector<float>>* matrix, const std::vector<float>* vec) {
	for (unsigned int i = 0; i < matrix->size(); i++) {
		for (unsigned int j = 0; j < (*matrix)[i].size(); j++) {
			(*matrix)[i][j] *= (*vec)[i];
		}
	}
};
float MathCNN::distance(std::vector<float>* p1, std::vector<float>* p2) {
	float sum = 0;
	for (int i = 0; i < p1->size(); i++) {
		sum += pow((*p1)[i] - (*p2)[i], 2);
	}
	return sqrt(sum);
};
void MathCNN::ReLU(std::vector<std::vector<float>>* z) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			if ((*z)[i][j] < 0) (*z)[i][j] = 0;
		}
	}
};
static std::vector<float> c = { 1 };
float MathCNN::ck(float k) {
	if (k == 0) return c[k];
	c.push_back(0);
	float sum = 0;
	for (int m = 0; m < k; m++) {
		float tmp = c[m] * c[k - 1 - m];
		tmp /= (m + 1) * (2 * m + 1);
		sum += tmp;
	}
	c[k] = sum;
	return c[k];
};
float MathCNN::erf_1(float x) {
	const float PI = 3.141592741;
	float sum = 0;
	for(int k=0; k<12; k++) {
		float tmp = ck(k) * pow(sqrt(PI) * 0.5 * x, 2 * k + 1);
		tmp /= 2 * k + 1;
		sum += tmp;
	}
	return sum;	
};
float MathCNN::getDistance(float sigmaW, float u) {
	return pow(((sigmaW * sqrt(2) * erf_1(2 * u - 1) - 5) / 2), 2) - 0.1;
};
int MathCNN::factorial(int n) {
	if (n == 0) return 1;
	int tmp = n * factorial(n - 1);
	return n * factorial(n - 1);
};
float MathCNN::getMeanDistance(float s) {
	const float E = 2.718281828;
	float m = 0.561756 * pow(E, -9.537722 * pow(s, -2));
	m -= 1.3187 * pow(E, -5.19039 * pow(s, -2));
	m *= s;
	m += (5 * pow(s, 2) + 123) * (erf(2.27824 / s) - erf(-3.08832 / s)) / 40;
	return m;
};
std::vector<float> MathCNN::dotProduct(const std::vector<float>* in1, const std::vector<float>* in2) {
	std::vector<float> result;
	for (unsigned int i = 0; i < in1->size(); i++) {
		result.push_back((*in1)[i] * (*in2)[i]);
	}
	return result;
};
std::vector<float> MathCNN::subtractVec(std::vector<float>* vec1, std::vector<float>* vec2) {
	std::vector<float> result;
	if (vec1->size() != vec2->size()) return result;
	for (unsigned int i = 0; i < vec1->size(); i++) {
		result.push_back((*vec1)[i] - (*vec2)[i]);
	}
	return result;
};
int MathCNN::convSize(int x, int s, int p, int k) {
	return ((x + 2 * p - k) / s) + 1;
};
featureMap MathCNN::conv(convAL* layer, convA* kernel, int stride) {
	//conv: BxHxT
	int kernelS = kernel->size();
	int resWidth = MathCNN::convSize(layer->size(), stride, 0, kernelS);
	int resHeight = MathCNN::convSize((*layer)[0].size(), stride, 0, kernelS);
	int featureMaps = (*layer)[0][0].size();
	featureMap result = featureMap(resWidth, std::vector<float>(resHeight));
	for (int i = 0; i < resWidth; i++) {
		for (int j = 0; j < resHeight; j++) {
			float sum = 0;
			for (int k1 = 0; k1 < kernelS; k1++) {
				for (int k2 = 0; k2 < kernelS; k2++) {
					for (int f = 0; f < featureMaps; f++) {
						int I = k1 + i * stride,
							J = k2 + j * stride;
						sum += (*kernel)[k1][k2][f] * (*layer)[I][J][f];
					}
				}
			}
			result[i][j] = sum;
		}
	}
	return result;
};
featureMap MathCNN::conv(convAL* layer, convA* kernel, int stride, int padding) {
	convA newL = *layer;//aStd(layer);
	int featureMaps = (*layer)[0][0].size();
	std::vector<float> border2(featureMaps);
	for (int i = 0; i < newL.size(); i++) { //width
		for (int j = 0; j < padding; j++) {
			newL[i].insert(newL[i].begin(), border2);
			newL[i].push_back(border2);
		}
	}
	std::vector<std::vector<float>> border(newL[0].size(), std::vector<float>(featureMaps));
	for (int j = 0; j < padding; j++) {
		newL.insert(newL.begin(), border);
		newL.push_back(border);
	}
	return conv(&newL, kernel, stride);
};/*
featureMap MathCNN::conv(convA* layer, convA* kernel, int stride) {
	//conv: BxHxT
	int kernelS = kernel->size();
	int resWidth = MathCNN::convSize(layer->size(), stride, 0, kernelS);
	int resHeight = MathCNN::convSize((*layer)[0].size(), stride, 0, kernelS);
	int featureMaps = (*layer)[0][0].size();
	featureMap result = featureMap(resWidth, std::vector<float>(resHeight));
	for (int i = 0; i < resWidth; i++) {
		for (int j = 0; j < resHeight; j++) {
			float sum = 0;
			for (int k1 = 0; k1 < kernelS; k1++) {
				for (int k2 = 0; k2 < kernelS; k2++) {
					for (int f = 0; f < featureMaps; f++) {
						int I = k1 + i * stride,
							J = k2 + j * stride;
						sum += (*kernel)[k1][k2][f] * (*layer)[I][J][f];
					}
				}
			}
			result[i][j] = sum;
		}
	}
	return result;
};
featureMap MathCNN::conv(convA* layer, convA* kernel, int stride, int padding) {
	convA newL = *layer;
	int featureMaps = (*layer)[0][0].size();
	std::vector<float> border2(featureMaps);
	for (int i = 0; i < newL.size(); i++) { //width
		for (int j = 0; j < padding; j++) {
			newL[i].insert(newL[i].begin(), border2);
			newL[i].push_back(border2);
		}
	}
	std::vector<std::vector<float>> border(newL[0].size(), std::vector<float>(featureMaps));
	for (int j = 0; j < padding; j++) {
		newL.insert(newL.begin(), border);
		newL.push_back(border);
	}
	convAL newLx = asStxxl(&newL);
	return conv(&newLx, kernel, stride);
};*/
std::vector<convA> MathCNN::ReLU(std::vector<convA>* z) {
	std::vector<convA> result = *z;
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					if ((*z)[i][j][k][l] > 0) {
						result[i][j][k][l] = (*z)[i][j][k][l];
					} else {
						result[i][j][k][l] = 0;
					}
				}
			}
		}
	}
	return result;
};
std::vector<convA> MathCNN::ReLU_(std::vector<convA>* z) {
	std::vector<convA> result = *z;
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					if ((*z)[i][j][k][l] > 0) {
						result[i][j][k][l] = 1;
					}
					else {
						result[i][j][k][l] = 0;
					}
				}
			}
		}
	}
	return result;
};
void MathCNN::ReLU(vconvAL* z, vconvAL* writeInto) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					if ((*z)[i][j][k][l] > 0) {
						(*writeInto)[i][j][k][l] = (*z)[i][j][k][l];
					}
					else {
						(*writeInto)[i][j][k][l] = 0;
					}
				}
			}
		}
	}
};
void MathCNN::ReLU_(vconvAL* z, vconvAL* writeInto) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					if ((*z)[i][j][k][l] > 0) {
						(*writeInto)[i][j][k][l] = 1;
					}
					else {
						(*writeInto)[i][j][k][l] = 0;
					}
				}
			}
		}
	}
};
void MathCNN::multiplyElementwise(std::vector<convA>* z, convA* gamma) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					(*z)[i][j][k][l] *= (*gamma)[j][k][l];
				}
			}
		}
	}
};
void MathCNN::addElementwise(std::vector<convA>* z, convA* beta) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					(*z)[i][j][k][l] += (*beta)[j][k][l];
				}
			}
		}
	}
};
void MathCNN::retransformBN(vconvAL* z, std::vector<float>* gamma, std::vector<float>* beta) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					(*z)[i][j][k][l] *= (*gamma)[l];
					(*z)[i][j][k][l] += (*beta)[l];
				}
			}
		}
	}
};
void MathCNN::addActMapwise(vconvAL* z, std::vector<float>* beta) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					(*z)[i][j][k][l] += (*beta)[l];
				}
			}
		}
	}
};
convAL MathCNN::dotProduct(convAL* mtrx1, convAL* mtrx2) {
	convAL result;
	for (unsigned int i = 0; i < mtrx1->size(); i++) {
		result.push_back(vectorff());
		for (unsigned int j = 0; j < (*mtrx1)[i].size(); j++) {
			result[i].push_back(vectorf());
			for (unsigned int k = 0; k < (*mtrx1)[i][j].size(); k++) {
				result[i][j].push_back((*mtrx1)[i][j][k] * (*mtrx2)[i][j][k]);
			}
		}
	}
	return result;
};
convA MathCNN::rotate180(convA* kernel) {
	convA result = *kernel;
	for (unsigned int w = 0; w < kernel->size(); w++) {
		for (unsigned int h = 0; h < (*kernel)[w].size(); h++) {
			int W = (*kernel).size() - w - 1,
				H = (*kernel)[w].size() - h - 1;
			for (unsigned int d = 0; d < (*kernel)[w][h].size(); d++) {
				//int D = (*kernel)[w][h].size() - d - 1;
				result[W][H][d] = (*kernel)[w][h][d];
			}
		}
	}
	return result;
};
convAL MathCNN::asStxxl(convA* in) {
	//convAL ret = convAL(in->size(), vectorff((*in)[0].size(), vectorf((*in)[0][0].size())));
	convAL ret;
	for (unsigned int i = 0; i < in->size(); i++) {
		ret.push_back(vectorff());
		for (unsigned int j = 0; j < (*in)[i].size(); j++) {
			ret[i].push_back(vectorf());
			for (unsigned int k = 0; k < (*in)[i][j].size(); k++) {
				ret[i][j].push_back((*in)[i][j][k]);
			}
		}
	}
	return ret;
};
vconvAL MathCNN::asStxxl(std::vector<convA>* in) {
	vconvAL ret;
	for (unsigned int i = 0; i < in->size(); i++) {
		//ret.push_back(MathCNN::asStxxl(&(*in)[i]));
		ret.push_back((*in)[i]);
	}
	return ret;
};
convA MathCNN::aStd(convAL* in) {
	convA ret;
	for (unsigned int i = 0; i < in->size(); i++) {
		ret.push_back(denseW());
		for (unsigned int j = 0; j < (*in)[i].size(); j++) {
			ret[j].push_back(denseA());
			for (unsigned int k = 0; k < (*in)[i][j].size(); k++) {
				ret[i][j].push_back((*in)[i][j][k]);
			}
		}
	}
	return ret;
};