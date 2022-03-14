#include "Dense.h"
#include "MathCNN.h"
#include <random>
#include <thread>
#include <iostream>

Dense::Dense(int size_, std::vector<denseA>* input_) : size(size_), input(input_) {
	sl_1 = (*input_)[0].size();
	activations = std::vector<denseA>(1, denseA(size));	//denseA(size);
	gradient = std::vector<denseA>(1, denseA(sl_1));		//denseA(sl_1);
	upGradient = nullptr;
	avgM = denseA(size);
	avgVar = denseA(size);
};
void Dense::forward() {
	std::vector<denseA> weightsT = MathCNN::transposeM(&weights);
	std::vector<std::vector<float>> z = MathCNN::MMProduct(input, &weightsT);
	zws = z;

	if (!lastLayer) {
		if (!converged) {
			denseA mean = denseA(zws[0].size()),
				   stddev2 = denseA(zws[0].size());
			BatchNorm(&z, &mean, &stddev2, delta);
			m = mean;
			var = stddev2;
			avgM = MathCNN::addVec(&avgM, &m);
			avgVar = MathCNN::addVec(&avgVar, &var);
		} else {
			BatchNorm(&z, avgM, avgVar, delta);
		}
		u = z;
		z = MathCNN::transposeM(&z);
		MathCNN::MVmultiplyColumn(&z, &gamma);
		MathCNN::MVaddColumn(&z, &beta);
		zws_hat = MathCNN::transposeM(&z);
	} else {
		z = MathCNN::transposeM(&z);
		MathCNN::MVaddColumn(&z, &beta);
	}
	activations = MathCNN::transposeM(&z);
	if(!lastLayer) MathCNN::ReLU(&activations);
};
void Dense::backward() {
	std::vector<denseA> d_zh = *upGradient;
	std::vector<denseA> dz;
	dz.reserve(d_zh.size());
	if (!lastLayer) {
		for (unsigned int i = 0; i < d_zh.size(); i++) {
			denseA z_h = zws_hat[i];
			MathCNN::ReLU_(&z_h);
			d_zh[i] = MathCNN::dotProduct(&z_h, &((*upGradient)[i]));
		}
		std::vector<float> d_beta;
		std::vector<float> d_gamma;
		std::vector<denseA> d_u;
		d_u.reserve(u.size());
		d_u = std::vector<denseA>(u.size(), denseA(u[0].size()));
		d_beta.reserve(d_zh[0].size());
		d_gamma.reserve(d_zh[0].size());
		denseA d_var = denseA(d_zh[0].size());
		for (unsigned int j = 0; j < d_zh[0].size(); j++) { //this Layer
			float avg_dB = 0,
				  avg_dG = 0;
			for (unsigned int i = 0; i < d_zh.size(); i++) { //batches
				avg_dB += d_zh[i][j];
				avg_dG += d_zh[i][j] * u[i][j];
				d_u[i][j] = d_zh[i][j] * gamma[j];
				d_var[j] += d_u[i][j] * (m[j] - zws[i][j]);
			}
			d_var[j] /= 2 * sqrt(pow(var[j] + delta, 3));
			avg_dB /= (float) d_zh.size();
			avg_dG /= (float) d_zh.size();
			d_beta.emplace_back(avg_dB);
			d_gamma.emplace_back(avg_dG);
		}
		int batchS = zws.size();
		betaGradients = d_beta;
		gammaGradients = d_gamma;
		denseA d_mean = denseA(zws[0].size());
		for (unsigned int j = 0; j < zws[0].size(); j++) { //this Layer
			float tmp1 = 0, tmp2 = 0;
			for (unsigned int i = 0; i < zws.size(); i++) { //batches
				tmp1 += m[j] - zws[i][j];
				tmp2 += d_u[i][j];
			}
			d_mean[j] = d_var[j] * 2 * tmp1 / ((float) batchS);
			d_mean[j] -= tmp2 / sqrt(var[j] + delta);
		}
		std::vector<denseA> d_z(zws.size(), denseA(zws[0].size()));
		for (unsigned int i = 0; i < d_z.size(); i++) { //batches
			for (unsigned int j = 0; j < d_z[i].size(); j++) { //this Layer
				d_z[i][j] = d_mean[j] / ((float) batchS);
				d_z[i][j] += d_var[j] * 2 * (zws[i][j] - m[j]) / ((float) batchS);
				d_z[i][j] += d_u[i][j] / sqrt(var[j] + delta);
			}
		}
		dz = d_z;
	} else {
		dz = d_zh;
		betaGradients = std::vector<float>();
		betaGradients.reserve(dz[0].size());
		for (unsigned int i = 0; i < dz[0].size(); i++) {
			float avgB = 0;
			for (unsigned int j = 0; j < dz.size(); j++) {
				avgB += dz[j][i];
			}
			avgB /= (float) dz.size();
			betaGradients.emplace_back(avgB);
		}
	}
	weightGradients = denseW();
	weightGradients.reserve(weights.size());
	weightGradients = denseW(weights.size(), std::vector<float>(weights[0].size()));
	denseW& weights_ = weights;
	denseW& weightGradients_ = weightGradients;
	std::vector<denseA>* input_ = input;
	std::vector<denseA>& zws_ = zws;
	std::vector<denseA>& dz_ = dz;
	std::thread t = std::thread([&weights_, &weightGradients_, &zws_, &dz_, input_]() {
		for (unsigned int i = 0; i < weights_.size(); i++) {
			for (unsigned int j = 0; j < weights_[i].size(); j++) {
				for (unsigned int k = 0; k < zws_.size(); k++) {
					weightGradients_[i][j] += dz_[k][i] * (*input_)[k][j];
				}
				weightGradients_[i][j] /= zws_.size();
			}
		}
	});
	if (gradient.size() != input->size()) {
		gradient.reserve(input->size());
		gradient = std::vector<denseA>(input->size(), std::vector<float>((*input)[0].size()));
	}
	//vector<denseA> G(input->size(), vector<float>((*input)[0].size(), 0));
	for (unsigned int i = 0; i < gradient.size(); i++) { //i=batches
		for (unsigned int j = 0; j < gradient[i].size(); j++) { //j=prevLayer
			gradient[i][j] = 0;
			for (unsigned int k = 0; k < zws[0].size(); k++) { //k=thisLayer
				gradient[i][j] += weights[k][j] * dz[i][k];
			}
		}
	}
	//gradient = G;
	t.join();
	updateParameters();
};
void Dense::initialiseParameters(std::string act) {
	float mean = (float) 0;
	float dev = 1;
	if (act == "ReLU") {
		dev = sqrt((float)2 / (float)sl_1); //He initialization
	}
	std::default_random_engine generator;
	//weightVelocity1 = vector<vector<float>>(weights.size(), vector<float>(weights[0].size()));
	//weightVelocity2 = vector<vector<float>>(weights.size(), vector<float>(weights[0].size()));
	weights.reserve(size);
	weightVelocity1.reserve(size);
	weightVelocity2.reserve(size);
	for (int i = 0; i < size; i++) {
		weights.emplace_back(std::vector<float>());
		weightVelocity1.emplace_back(std::vector<float>());
		weightVelocity2.emplace_back(std::vector<float>());
		weights[i].reserve(sl_1);
		weightVelocity1[i].reserve(sl_1);
		weightVelocity2[i].reserve(sl_1);
		for (int j = 0; j < sl_1; j++) {
			weights[i].emplace_back(MathCNN::getRandomNormal(&mean, &dev, &generator));
			weightVelocity1[i].emplace_back(0);
			weightVelocity2[i].emplace_back(0);
		}
	}
	beta.reserve(size);
	gamma.reserve(size);
	for (int i = 0; i < size; i++) {
		beta.emplace_back((float)0);
		gamma.emplace_back((float)1);
	}
	gammaVelocity1.reserve(size);
	gammaVelocity2.reserve(size);
	betaVelocity1.reserve(size);
	betaVelocity2.reserve(size);
	gammaVelocity1 = std::vector<float>(gamma.size());
	gammaVelocity2 = std::vector<float>(gamma.size());
	betaVelocity1 = std::vector<float>(beta.size());
	betaVelocity2 = std::vector<float>(beta.size());
};
void Dense::setUpGradient(std::vector<denseA>* upGradient_) {
	upGradient = upGradient_;
};
void Dense::updateParameters() {
	if (!lastLayer) {
		for (unsigned int i = 0; i < beta.size(); i++) {
			betaVelocity1[i] = beta_o1 * betaVelocity1[i] + ((1 - beta_o1) * betaGradients[i]);
			betaVelocity2[i] = beta_o2 * betaVelocity2[i] + ((1 - beta_o2) * pow(betaGradients[i], 2));
			gammaVelocity1[i] = beta_o1 * gammaVelocity1[i] + ((1 - beta_o1) * gammaGradients[i]);
			gammaVelocity2[i] = beta_o2 * gammaVelocity2[i] + ((1 - beta_o2) * pow(gammaGradients[i], 2));
		}
		for (unsigned int i = 0; i < beta.size(); i++) {
			float betaV1_h = betaVelocity1[i] / (1 - pow(beta_o1, epoch));
			float betaV2_h = betaVelocity2[i] / (1 - pow(beta_o2, epoch));
			float gammaV1_h = gammaVelocity1[i] / (1 - pow(beta_o1, epoch));
			float gammaV2_h = gammaVelocity2[i] / (1 - pow(beta_o2, epoch));
			beta[i] -= eta * (betaV1_h / sqrt(betaV2_h + 0.00001));
			//beta[i] -= eta * betaGradients[i];
			gamma[i] -= eta * (gammaV1_h / sqrt(gammaV2_h + 0.00001));
			//gamma[i] -= eta * gammaGradients[i];
		}
	} else { //lastLayer
		for (unsigned int i = 0; i < beta.size(); i++) {
			betaVelocity1[i] = beta_o1 * betaVelocity1[i] + ((1 - beta_o1) * betaGradients[i]);
			betaVelocity2[i] = beta_o2 * betaVelocity2[i] + ((1 - beta_o2) * pow(betaGradients[i], 2));
		}
		for (unsigned int i = 0; i < beta.size(); i++) {
			float betaV1_h = betaVelocity1[i] / (1 - pow(beta_o1, epoch));
			float betaV2_h = betaVelocity2[i] / (1 - pow(beta_o2, epoch));
			beta[i] -= eta * (betaV1_h / sqrt(betaV2_h + 0.00001));
			//beta[i] -= eta * betaGradients[i];
		}
	}
	for (unsigned int i = 0; i < weights.size(); i++) {
		for (unsigned int j = 0; j < weights[i].size(); j++) {
			weightVelocity1[i][j] = beta_o1 * weightVelocity1[i][j] + ((1 - beta_o1) * weightGradients[i][j]);
			weightVelocity2[i][j] = beta_o2 * weightVelocity2[i][j] + ((1 - beta_o2) * pow(weightGradients[i][j], 2));
		}
	}
	for (unsigned int i = 0; i < weights.size(); i++) {
		for (unsigned int j = 0; j < weights[i].size(); j++) {
			float weightV1_h = weightVelocity1[i][j] / (1 - pow(beta_o1, epoch));
			float weightV2_h = weightVelocity2[i][j] / (1 - pow(beta_o2, epoch));
			weights[i][j] -= eta * (weightV1_h / sqrt(weightV2_h + 0.00001));
			//weights[i][j] -= eta * weightGradients[i][j];
		}
	}
	epoch++;
}
void Dense::endTraining() {
	converged = true;
	/*float m = zws.size();
	for (unsigned int i = 0; i < zws[0].size(); i++) {
		avgM[i] /= (epoch - 1);
		avgVar[i] *= m / ((m - 1) * (epoch - 1));
	}*/
	betaGradients.clear();
	betaGradients.shrink_to_fit();
	betaVelocity1.clear();
	betaVelocity1.shrink_to_fit();
	betaVelocity2.clear();
	betaVelocity2.shrink_to_fit();
	gammaGradients.clear();
	gammaGradients.shrink_to_fit();
	gammaVelocity1.clear();
	gammaVelocity1.shrink_to_fit();
	gammaVelocity2.clear();
	gammaVelocity2.shrink_to_fit();
	weightGradients.clear();
	weightGradients.shrink_to_fit();
	weightVelocity1.clear();
	weightVelocity1.shrink_to_fit();
	weightVelocity2.clear();
	weightVelocity2.shrink_to_fit();
	gradient.clear();
	gradient.shrink_to_fit();
	activations.clear();
	activations.shrink_to_fit();
}

denseA Dense::getAvgMean() {
	denseA avg = denseA(avgM);
	for (unsigned int i = 0; i < zws[0].size(); i++) {
		avg[i] /= (epoch - 1);
	}
	return avg;
}

denseA Dense::getAvgVar() {
	float m = zws.size();
	denseA avg = denseA(avgVar);
	for (unsigned int i = 0; i < zws[0].size(); i++) {
		avg[i] *= m / ((m - 1) * (epoch - 1));
	}
	return avg;
}