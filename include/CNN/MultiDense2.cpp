#include "MultiDense2.h"
#include "MathCNN.h"
#include <math.h>
#include <iostream>
#include <random>
#include <time.h>

MultiDense2::MultiDense2(int size_, std::vector<denseA>* inputAct_, std::vector<std::vector<float>>* inputPos_, float* m, int ixL) : size(size_), inputAct(inputAct_), inputPos(inputPos_), meanl_1(m), ixLayer(ixL) {
	sl_1 = (*inputAct)[0].size();
	activations = std::vector<denseA>(1, denseA(size));
	gradient = std::vector<denseA>(1, denseA(size));
	position = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
	upGradient = nullptr;
	avgM = denseA(size);
	avgVar = denseA(size);
};
void MultiDense2::forward() {
	std::vector<denseA> z = std::vector<denseA>(inputAct->size(), denseA(size));
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < inputAct->size(); j++) {
			for (unsigned int k = 0; k < (*inputAct)[j].size(); k++) {
				std::vector<float>* inPos = &((*inputPos)[k]),
								  * outPos = &(position[i]);
				float distance = MathCNN::distance(inPos, outPos);
				float weight = alphaM[i] * distance + betaM[i];
				z[j][i] += weight * (*inputAct)[j][k];
			}
		}
	}
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
		}
		else {
			BatchNorm(&z, avgM, avgVar, delta);
		}
		u = z;
		z = MathCNN::transposeM(&z);
		MathCNN::MVmultiplyColumn(&z, &gamma);
		MathCNN::MVaddColumn(&z, &beta);
		zws_hat = MathCNN::transposeM(&z);
	}
	else {
		z = MathCNN::transposeM(&z);
		MathCNN::MVaddColumn(&z, &beta);
	}
	activations = MathCNN::transposeM(&z);
	if (!lastLayer) MathCNN::ReLU(&activations);
};
void MultiDense2::backward() {
	std::vector<denseA> d_zh = *upGradient;
	std::vector<denseA> dz;
	if (!lastLayer) {
		for (unsigned int i = 0; i < d_zh.size(); i++) {
			denseA z_h = zws_hat[i];
			MathCNN::ReLU_(&z_h);
			d_zh[i] = MathCNN::dotProduct(&z_h, &((*upGradient)[i]));
		}
		std::vector<float> d_beta;
		std::vector<float> d_gamma;
		std::vector<denseA> d_u = u;
		denseA d_var = denseA(d_zh[0].size());
		for (unsigned int j = 0; j < d_zh[0].size(); j++) {
			float avg_dB = 0,
				avg_dG = 0;
			for (unsigned int i = 0; i < d_zh.size(); i++) {
				avg_dB += d_zh[i][j];
				avg_dG += d_zh[i][j] * u[i][j];
				d_u[i][j] = d_zh[i][j] * gamma[j];
				d_var[j] += d_u[i][j] * (m[j] - zws[i][j]);
			}
			d_var[j] /= 2 * sqrt(pow(var[j] + delta, 3));
			avg_dB /= (float)d_zh.size();
			avg_dG /= (float)d_zh.size();
			d_beta.push_back(avg_dB);
			d_gamma.push_back(avg_dG);
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
			d_mean[j] = d_var[j] * 2 * tmp1 / ((float)batchS);
			d_mean[j] -= tmp2 / sqrt(var[j] + delta);
		}
		std::vector<denseA> d_z(zws.size(), denseA(zws[0].size()));
		for (unsigned int i = 0; i < d_z.size(); i++) {
			for (unsigned int j = 0; j < d_z[i].size(); j++) {
				d_z[i][j] = d_mean[j] / ((float)batchS);
				d_z[i][j] += d_var[j] * 2 * (zws[i][j] - m[j]) / (float)batchS;
				d_z[i][j] += d_u[i][j] / sqrt(var[j] + delta);
			}
		}
		dz = d_z;
	}
	else {
		dz = d_zh;
		betaGradients = std::vector<float>(0);
		for (unsigned int i = 0; i < dz[0].size(); i++) {
			float avgB = 0;
			for (unsigned int j = 0; j < dz.size(); j++) {
				avgB += dz[j][i];
			}
			avgB /= dz.size();
			betaGradients.push_back(avgB);
		}
	}
	std::vector<denseW> weightGradients = std::vector<denseW>(zws.size(), denseW(size, std::vector<float>(sl_1, 0)));
	for (unsigned int i = 0; i < weightGradients.size(); i++) { //i=batches
		for (unsigned int j = 0; j < weightGradients[i].size(); j++) { //j=thisLayer
			for (unsigned int k = 0; k < weightGradients[i][j].size(); k++) { //k=prevLayer
				weightGradients[i][j][k] = dz[i][j] * (*inputAct)[i][k];
			}
		}
	}
	wG = weightGradients;
	std::vector<denseA> gra(inputAct->size(), std::vector<float>((*inputAct)[0].size(), 0));
	for (unsigned int i = 0; i < gra.size(); i++) { //i=batches
		for (unsigned int j = 0; j < gra[i].size(); j++) { //j=prevLayer
			for (unsigned int k = 0; k < zws[0].size(); k++) { //k=thisLayer
				std::vector<float>* inPos = &((*inputPos)[j]),
					* outPos = &(position[k]);
				float distance = MathCNN::distance(inPos, outPos);
				float weight = alphaM[k] * distance + betaM[k];
				gra[i][j] += weight * dz[i][k];
			}
		}
	}
	gradient = gra;
	//own backprop
	betaMGradients = std::vector<float>(size);
	alphaMGradients = std::vector<float>(size);
	for (unsigned int i = 0; i < betaMGradients.size(); i++) { //thisLayer
		float avgB = 0;
		float avgA = 0;
		for (unsigned int k = 0; k < zws.size(); k++) { //batches
			float sumB = 0;
			float sumA = 0;
			for (unsigned int j = 0; j < sl_1; j++) { //prevLayer
				sumB += weightGradients[k][i][j];
				std::vector<float>* inPos = &((*inputPos)[j]),
								  * outPos = &(position[i]);
				float distance = MathCNN::distance(inPos, outPos);
				sumA += weightGradients[k][i][j] * distance;
			}
			avgB += sumB;
			avgA += sumA;
		}
		avgB /= zws.size();
		avgA /= zws.size();
		betaMGradients[i] = avgB;
		alphaMGradients[i] = avgA;
	}
	for (unsigned int i = 0; i < size; i++) { //this Layer
		for (unsigned int j = 0; j < dimensions; j++) { //dimensions
			float avgC = 0;
			for (unsigned int k = 0; k < zws.size(); k++) { //batches
				float sum = 0;
				for (unsigned int l = 0; l < sl_1; l++) { //prev Layer
					float tmp = weightGradients[k][i][l];
					std::vector<float>* inPos = &((*inputPos)[l]),
									  * outPos = &(position[i]);
					float distance = MathCNN::distance(inPos, outPos);
					tmp *= alphaM[i];
					tmp *= (position[i][j] - (*inputPos)[l][j]) / distance;
					sum += tmp;
				}
				avgC += sum;
			}
			avgC /= zws.size();
			positionGradients[i][j] = avgC;
		}
	}
	if (!lastLayer && nextW) {
		for (unsigned int i = 0; i < size; i++) { //this Layer
			for (unsigned int j = 0; j < dimensions; j++) { //dimensions
				float avgC = 0;
				for (unsigned int k = 0; k < zws.size(); k++) { //batches
					float sum = 0;
					for (unsigned int l = 0; l < posp1->size(); l++) { //next Layer
						float tmp = (*wGp1)[k][l][i];
						std::vector<float>* inPos = &(position[i]),
							* outPos = &((*posp1)[l]);
						float distance = MathCNN::distance(inPos, outPos);
						tmp *= (*ap1)[l];
						tmp *= (position[i][j] - (*posp1)[l][j]) / distance;
						sum += tmp;
					}
					avgC += sum;
				}
				avgC /= zws.size();
				positionGradients[i][j] += avgC;
			}
		}
	}
	updateParameters();
};
void MultiDense2::initialiseParameters(std::string actF) {
	for (int i = 0; i < size; i++) {
		alphaM.push_back((float)1);
		betaM.push_back((float)-1 * meanDistance); //-alpha*distance
		gamma.push_back((float)1);
		beta.push_back((float)0);
	}
	float stddev = 0;
	if (actF == "ReLU") {
		stddev = 1 / (1 * sqrt((float)sl_1)); // 1/alpha*sqrt(n)
	}
	//mean = *meanl_1 + (MathCNN::getMeanDistance(stddev) / sqrt(2));
	srand((unsigned)time(NULL));
	const float PI = 3.141592741;
	std::default_random_engine generator;
	for (unsigned int i = 0; i < position.size(); i++) {
		if (ixLayer % 2 != 0) {
			float distance = MathCNN::getRandomNormal(&meanDistance, &stddev, &generator);
			float u = (float)rand() / RAND_MAX;
			float phi = 2 * PI * u;
			u = (float)rand() / RAND_MAX;
			float theta = 2 * PI * u;
			position[i][0] = distance * cos(phi) * cos(theta);
			position[i][1] = distance * cos(phi) * sin(theta);
			position[i][2] = -1 * distance * sin(phi);
		}
		else {
			float mean = 0;
			position[i][0] = MathCNN::getRandomNormal(&mean, &stddev, &generator);
			position[i][1] = MathCNN::getRandomNormal(&mean, &stddev, &generator);
			position[i][2] = MathCNN::getRandomNormal(&mean, &stddev, &generator);
		}
	}

	betaVelocity1 = std::vector<float>(beta.size());
	betaVelocity2 = std::vector<float>(beta.size());
	gammaVelocity1 = std::vector<float>(gamma.size());
	gammaVelocity2 = std::vector<float>(gamma.size());
	betaMVelocity1 = std::vector<float>(betaM.size());
	betaMVelocity2 = std::vector<float>(betaM.size());
	alphaMVelocity1 = std::vector<float>(alphaM.size());
	alphaMVelocity2 = std::vector<float>(alphaM.size());
	positionVelocity1 = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
	positionVelocity2 = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
	positionGradients = std::vector<std::vector<float>>(size, std::vector<float>(dimensions));
};
void MultiDense2::setUpGradient(std::vector<denseA>* upGradient_) {
	upGradient = upGradient_;
};
void MultiDense2::setInputPosition(std::vector<std::vector<float>>* inputPos_) {
	inputPos = inputPos_;
};
void MultiDense2::updateParameters() {
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
	}
	else { //lastLayer
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
	for (unsigned int i = 0; i < alphaMGradients.size(); i++) {
		betaMVelocity1[i] = beta_o1 * betaMVelocity1[i] + ((1 - beta_o1) * betaMGradients[i]);
		betaMVelocity2[i] = beta_o2 * betaMVelocity2[i] + ((1 - beta_o2) * pow(betaMGradients[i], 2));
		alphaMVelocity1[i] = beta_o1 * alphaMVelocity1[i] + ((1 - beta_o1) * alphaMGradients[i]);
		alphaMVelocity2[i] = beta_o2 * alphaMVelocity2[i] + ((1 - beta_o2) * pow(alphaMGradients[i], 2));
	}
	for (unsigned int i = 0; i < alphaMGradients.size(); i++) {
		float betaMV1_h = betaMVelocity1[i] / (1 - pow(beta_o1, epoch));
		float betaMV2_h = betaMVelocity2[i] / (1 - pow(beta_o2, epoch));
		float alphaMV1_h = alphaMVelocity1[i] / (1 - pow(beta_o1, epoch));
		float alphaMV2_h = alphaMVelocity2[i] / (1 - pow(beta_o2, epoch));
		betaM[i] -= eta * (betaMV1_h / sqrt(betaMV2_h + 0.00001));
		alphaM[i] -= eta * (alphaMV1_h / sqrt(alphaMV2_h + 0.00001));
	}
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
	epoch++;
};
void MultiDense2::endTraining() {
	converged = true;
	/*float m = zws.size();
	avgM /= epoch;
	float m = zws.size() * zws[0].size();
	avgVar *= m / ((m - 1) * epoch);*/
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
	betaMGradients.clear();
	betaMGradients.shrink_to_fit();
	betaMVelocity1.clear();
	betaMVelocity1.shrink_to_fit();
	betaMVelocity2.clear();
	betaMVelocity2.shrink_to_fit();
	alphaMGradients.clear();
	alphaMGradients.shrink_to_fit();
	alphaMVelocity1.clear();
	alphaMVelocity1.shrink_to_fit();
	alphaMVelocity2.clear();
	alphaMVelocity2.shrink_to_fit();
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

denseA MultiDense2::getAvgMean() {
	denseA avg = denseA(avgM);
	for (unsigned int i = 0; i < zws[0].size(); i++) {
		avg[i] /= (epoch - 1);
	}
	return avg;
}

denseA MultiDense2::getAvgVar() {
	float m = zws.size();
	denseA avg = denseA(avgVar);
	for (unsigned int i = 0; i < zws[0].size(); i++) {
		avg[i] *= m / ((m - 1) * (epoch - 1));
	}
	return avg;
}