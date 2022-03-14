#include "Conv.h"
#include <math.h>
#include <iostream>
#include <thread>

Conv::Conv(int actMaps, vconvAL* input_, int kernelS_, int stride_, int padding_) : input(input_), stride(stride_), kernelS(kernelS_), padding(padding_) {
	sl_1[0] = (*input)[0].size();
	sl_1[1] = (*input)[0][0].size();
	sl_1[2] = (*input)[0][0][0].size();
	size[0] = MathCNN::convSize(sl_1[0], stride, padding, kernelS);
	size[1] = MathCNN::convSize(sl_1[1], stride, padding, kernelS);
	size[2] = actMaps;
	kernel.reserve(size[2]);
	for (unsigned int i = 0; i < size[2]; i++) {
		kernel.emplace_back(convA(kernelS, std::vector<std::vector<float>>(kernelS, std::vector<float>(sl_1[2]))));
	}
	layerDim.reserve(size[0]);
	for (unsigned int i = 0; i < size[0]; i++) {
		layerDim.emplace_back(std::vector<std::vector<float>>(size[1], std::vector<float>(size[2])));
	}
	//layerDimx = MathCNN::asStxxl(&layerDim);
	activations.push_back(layerDim);
	//activations = vector<convA>(1, convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2]))));		//convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2])));
	//gradient = vector<convA>(0);		//convA(sl_1[0], vector<vector<float>>(sl_1[1], vector<float>(sl_1[2])));
	upGradient = nullptr;
	//kernelThread = thread([]() {});
	//kernelThreads = vector<thread>(0);
	//gradientThreads = vector<thread>(0);
};
void Conv::forward() {
	vconvAL z;
	if (padding == 0) {
		std::vector<std::vector<int>> tDistribution;
		if (input->size() == 1) {
			tDistribution = { {0, 1} };
		}
		else {
			int cut = (int)input->size() / 4;
			tDistribution.push_back({ 0, cut });
			tDistribution.push_back({ cut, 2 * cut });
			tDistribution.push_back({ 2 * cut, 3 * cut });
			tDistribution.push_back({ 3 * cut, (int)input->size() });
		}	
		std::vector<std::thread> kernelThreads = std::vector<std::thread>(tDistribution.size());
		int(&size_)[3] = size;
		std::vector<convA>& kernel_ = kernel;
		vconvAL* input_ = input;
		vconvAL& z_ = z;
		convA& layerDim_ = layerDim;
		int& stride_ = stride;
		for(unsigned int i=0; i<input->size(); i++) {
			z_.push_back(layerDim);
		}
		for (unsigned int t = 0; t < tDistribution.size(); t++) {
			//kernelThreads[t] = thread([this, tDistribution, t, dz]() {
			kernelThreads[t] = std::thread([&size_, &kernel_, input_, &stride_, tDistribution, t, &z_, &layerDim_]() {
				for (unsigned int i = tDistribution[t][0]; i < tDistribution[t][1]; i++) {
					convA tmp;
					for (unsigned int j = 0; j < size_[2]; j++) { //actMaps 
						featureMap actMap = MathCNN::conv(&(*input_)[i], &kernel_[j], stride_);
						tmp.emplace_back(actMap);
					}
					convA tmp2 = layerDim_;
					for (unsigned int j = 0; j < size_[0]; j++) { //width
						for (unsigned int k = 0; k < size_[1]; k++) { //height
							for (unsigned int l = 0; l < size_[2]; l++) { //actMaps 
								tmp2[j][k][l] = tmp[l][j][k];
							}
						}
					}
					//z.push_back(MathCNN::asStxxl(&tmp2));
					//z_.push_back(tmp2);
					z_[i] = tmp2;
				}
			});
		}
		for (unsigned int b = 0; b < tDistribution.size(); b++) {
			kernelThreads[b].join();
		}/*
		for (unsigned int i = 0; i < input->size(); i++) { //batches
			convA tmp;
			for (unsigned int j = 0; j < size[2]; j++) { //actMaps 
				featureMap actMap = MathCNN::conv(&(*input)[i], &kernel[j], stride);
				tmp.emplace_back(actMap);
			}
			convA tmp2 = layerDim;
			for (unsigned int j = 0; j < size[0]; j++) { //width
				for (unsigned int k = 0; k < size[1]; k++) { //height
					for (unsigned int l = 0; l < size[2]; l++) { //actMaps 
						tmp2[j][k][l] = tmp[l][j][k];
					}
				}
			}
			//z.push_back(MathCNN::asStxxl(&tmp2));
			z.push_back(tmp2);
		}*/
	} else {
		for (unsigned int i = 0; i < input->size(); i++) { //batches
			convA tmp;
			tmp.reserve(size[2]);
			for (unsigned int j = 0; j < size[2]; j++) { //actMaps 
				featureMap actMap = MathCNN::conv(&(*input)[i], &kernel[j], stride, padding);
				tmp.emplace_back(actMap);
			}
			convA tmp2 = layerDim;
			for (unsigned int j = 0; j < size[0]; j++) { //width
				for (unsigned int k = 0; k < size[1]; k++) { //height
					for (unsigned int l = 0; l < size[2]; l++) { //actMaps 
						tmp2[j][k][l] = tmp[l][j][k];
					}
				}
			}
			//z.push_back(MathCNN::asStxxl(&tmp2));
			z.push_back(tmp2);
		}
	}
	zws = z;
	//zws = vector<convA>(z);
	if (BN) {
		if (!converged) {
			m = std::vector<float>(size[2]);
			var = std::vector<float>(size[2]);
			BatchNorm(&z, &m, &var, delta);
			avgM = MathCNN::addVec(&m, &avgM);
			avgVar = MathCNN::addVec(&var, &avgVar);
		} else {
			BatchNorm(&z, avgM, avgVar, delta);
		}
		u = z;
		MathCNN::retransformBN(&z, &gamma, &beta);
		zws_hat = z;
	} else {
		MathCNN::addActMapwise(&z, &biases);
	}
	if (activations.size() != input->size()) {
		activations = vconvAL();
		activations.reserve(input->size());
		for (unsigned int i = 0; i < input->size(); i++) {
			activations.push_back(layerDim);
		}
	}
	MathCNN::ReLU(&z, &activations);
};
void Conv::backward() {
	if (stride != 1) return;
	vconvAL relu_(zws_hat);
	if (!BN) relu_ = zws;
	MathCNN::ReLU_(&relu_, &relu_);
	vconvAL d_zh, dz;
	d_zh.reserve(input->size());
	for (unsigned int b = 0; b < input->size(); b++) { //batches
		d_zh.push_back(MathCNN::dotProduct(&relu_[b], &((*upGradient)[b])));
	}
	if (BN) {
		std::vector<float> d_beta(size[2]),
		                   d_gamma(size[2]);
		/*convA d_beta(size[0], vector<vector<float>>(size[1], vector<float>(size[2]))),
			  d_gamma(size[0], vector<vector<float>>(size[1], vector<float>(size[2])));*/
		vconvAL d_u;
		for (unsigned int i = 0; i < input->size(); i++) {
			d_u.push_back(layerDim);
		}
		for (unsigned int d = 0; d < size[2]; d++) {
			float avg_dB = 0,
				  avg_dG = 0;
			for (unsigned int w = 0; w < size[0]; w++) {
				for (unsigned int h = 0; h < size[1]; h++) {
					for (unsigned int b = 0; b < input->size(); b++) {
						avg_dB += d_zh[b][w][h][d];
						avg_dG += d_zh[b][w][h][d] * u[b][w][h][d];
					}
				}
			}
			avg_dB /= input->size();// * size[0] * size[1];
			avg_dG /= input->size();// * size[0] * size[1];
			d_beta[d] = avg_dB;
			d_gamma[d] = avg_dG;
		}
		for (unsigned int w = 0; w < size[0]; w++) {
			for (unsigned int h = 0; h < size[1]; h++) {
				for (unsigned int d = 0; d < size[2]; d++) {
					for (unsigned int b = 0; b < input->size(); b++) {
						d_u[b][w][h][d] = d_zh[b][w][h][d] * gamma[d];
					}
				}
			}
		}
		betaGradient = d_beta;
		gammaGradient = d_gamma;
		std::vector<float> d_var(size[2]);
		for (unsigned int d = 0; d < size[2]; d++) {
			for (unsigned int b = 0; b < input->size(); b++) {
				for (unsigned int w = 0; w < size[0]; w++) {
					for (unsigned int h = 0; h < size[1]; h++) {
						d_var[d] += d_u[b][w][h][d] * (m[d] - zws[b][w][h][d]);
					}
				}
			}
			d_var[d] /= 2 * sqrt(pow(var[d] + delta, 3));
		}
		std::vector<float> d_mean(size[2]);
		std::vector<float> tmp1(size[2]), tmp2(size[2]);
		int batchS = input->size() * size[0] * size[1];
		for (unsigned int d = 0; d < size[2]; d++) {
			for (unsigned int b = 0; b < input->size(); b++) {
				for (unsigned int w = 0; w < size[0]; w++) {
					for (unsigned int h = 0; h < size[1]; h++) {
						tmp1[d] += m[d] - zws[b][w][h][d];
						tmp2[d] += d_u[b][w][h][d];
					}
				}
			}
			d_mean[d] = d_var[d] * 2 * tmp1[d] / (float)batchS;
			d_mean[d] -= tmp2[d] / sqrt(var[d] + delta);
		}
		vconvAL d_z;
		d_z.reserve(input->size());
		for (unsigned int d = 0; d < size[2]; d++) {
			for (unsigned int b = 0; b < input->size(); b++) {
				if (d == 0) d_z.push_back(layerDim);
				for (unsigned int w = 0; w < size[0]; w++) {
					for (unsigned int h = 0; h < size[1]; h++) {
						d_z[b][w][h][d] = d_mean[d] / (float)batchS;
						d_z[b][w][h][d] += d_var[d] * 2 * (zws[b][w][h][d] - m[d]) / (float)batchS;
						d_z[b][w][h][d] += d_u[b][w][h][d] / sqrt(var[d] + delta);
					}
				}
			}
		}
		dz = d_z;
		//dz = vector<convA>(d_z);
	}
	else {
		dz = d_zh;
		//dz = vector<convA>(d_zh);
		biasesGradient = std::vector<float>();
		biasesGradient.reserve(size[2]);
		for (unsigned int d = 0; d < size[2]; d++) {
			float avgB = 0;
			for (unsigned int b = 0; b < input->size(); b++) {
				for (unsigned int w = 0; w < size[0]; w++) {
					for (unsigned int h = 0; h < size[1]; h++) {
						avgB += dz[b][w][h][d];
					}
				}
			}
			avgB /= (float)input->size();
			biasesGradient.emplace_back(avgB);
		}
	}
	try {
		std::vector<std::vector<int>> tDistribution;
		if (input->size() == 1) {
			tDistribution = { {0, 1} };
		}
		else {
			int cut = (int)input->size() / 4;
			tDistribution.push_back({ 0, cut });
			tDistribution.push_back({ cut, 2 * cut });
			tDistribution.push_back({ 2 * cut, 3 * cut });
			tDistribution.push_back({ 3 * cut, (int)input->size() });
		}
		//d_weights: 2d-conv w/ kernel = dz, ActMap = input
		kernelGradient = std::vector<convA>();
		kernelGradient.reserve(kernel.size());
		for (unsigned int i = 0; i < kernel.size(); i++) {
			kernelGradient.push_back(convA(kernel[0].size(), std::vector<std::vector<float>>(kernel[0][0].size(), std::vector<float>(kernel[0][0][0].size()))));
		}
		std::vector<std::thread> kernelThreads = std::vector<std::thread>(tDistribution.size());
		if (kernelThreads.size() == 0) {
			kernelThreads = std::vector<std::thread>(tDistribution.size());
		}
		int (&size_)[3] = size;
		int (&sl_1_)[3] = sl_1;
		std::vector<convA>& kernel_ = kernel;
		std::vector<convA>& kernelGradient_ = kernelGradient;
		vconvAL& gradient_ = gradient;
		vconvAL& dz_ = dz;
		vconvAL* input_ = input;
		int& padding_ = padding;
		int& stride_ = stride;
		for (unsigned int t = 0; t < tDistribution.size(); t++) {
			//kernelThreads[t] = thread([this, tDistribution, t, dz]() {
			kernelThreads[t] = std::thread([&size_, &sl_1_, &kernel_, &kernelGradient_, input_, &padding_, &stride_, tDistribution, t, &dz_]() {
				for (unsigned int b = tDistribution[t][0]; b < tDistribution[t][1]; b++) {
					for (unsigned int d = 0; d < size_[2]; d++) { //kernels = depth l
						for (unsigned int dk = 0; dk < kernel_[0][0][0].size(); dk++) { //kernel depth = depth l-1
							convA kernel2d(size_[0], std::vector<std::vector<float>>(size_[1], std::vector<float>(1)));
							for (unsigned int k1 = 0; k1 < kernel2d.size(); k1++) {
								for (unsigned int k2 = 0; k2 < kernel2d[k1].size(); k2++) {
									kernel2d[k1][k2][0] = dz_[b][k1][k2][d];
								}
							}
							convA actMap(sl_1_[0], std::vector<std::vector<float>>(sl_1_[1], std::vector<float>(1)));
							for (unsigned int w = 0; w < actMap.size(); w++) {
								for (unsigned int h = 0; h < actMap[w].size(); h++) {
									actMap[w][h][0] = (*input_)[b][w][h][dk];
								}
							}
							featureMap dw2d;
							if (padding_ == 0) {
								dw2d = MathCNN::conv(&actMap, &kernel2d, stride_);
							}
							else {
								dw2d = MathCNN::conv(&actMap, &kernel2d, stride_, padding_);
							}
							for (unsigned int wk = 0; wk < dw2d.size(); wk++) {
								for (unsigned int hk = 0; hk < dw2d.size(); hk++) {
									kernelGradient_[d][wk][hk][dk] += dw2d[wk][hk];
								}
							}
						}
					}
				}
			});
		}
		/*for (unsigned int b = 0; b < input->size(); b++) {
			kernelThreads[b] = thread([this, b, dz]() {
				for (unsigned int d = 0; d < size[2]; d++) { //kernels = depth l
					for (unsigned int dk = 0; dk < kernel[0][0][0].size(); dk++) { //kernel depth = depth l-1
						convA kernel2d(size[0], vector<vector<float>>(size[1], vector<float>(1)));
						for (unsigned int k1 = 0; k1 < kernel2d.size(); k1++) {
							for (unsigned int k2 = 0; k2 < kernel2d[k1].size(); k2++) {
								kernel2d[k1][k2][0] = dz[b][k1][k2][d];
							}
						}
						convA actMap(sl_1[0], vector<vector<float>>(sl_1[1], vector<float>(1)));
						for (unsigned int w = 0; w < actMap.size(); w++) {
							for (unsigned int h = 0; h < actMap[w].size(); h++) {
								actMap[w][h][0] = (*input)[b][w][h][dk];
							}
						}
						featureMap dw2d;
						if (padding == 0) {
							dw2d = MathCNN::conv(&actMap, &kernel2d, stride);
						}
						else {
							dw2d = MathCNN::conv(&actMap, &kernel2d, stride, padding);
						}
						for (unsigned int wk = 0; wk < dw2d.size(); wk++) {
							for (unsigned int hk = 0; hk < dw2d.size(); hk++) {
								kernelGradient[d][wk][hk][dk] += dw2d[wk][hk];
							}
						}
					}
				}
			});
		}*/
		/*thread kernelThread = thread([this, dz]() {
			for (unsigned int b = 0; b < input->size(); b++) {
				for (unsigned int d = 0; d < size[2]; d++) { //kernels = depth l
					for (unsigned int dk = 0; dk < kernel[0][0][0].size(); dk++) { //kernel depth = depth l-1
						convA kernel2d(size[0], vector<vector<float>>(size[1], vector<float>(1)));
						for (unsigned int k1 = 0; k1 < kernel2d.size(); k1++) {
							for (unsigned int k2 = 0; k2 < kernel2d[k1].size(); k2++) {
								kernel2d[k1][k2][0] = dz[b][k1][k2][d];
							}
						}
						convA actMap(sl_1[0], vector<vector<float>>(sl_1[1], vector<float>(1)));
						for (unsigned int w = 0; w < actMap.size(); w++) {
							for (unsigned int h = 0; h < actMap[w].size(); h++) {
								actMap[w][h][0] = (*input)[b][w][h][dk];
							}
						}
						featureMap dw2d;
						if (padding == 0) {
							dw2d = MathCNN::conv(&actMap, &kernel2d, stride);
						}
						else {
							dw2d = MathCNN::conv(&actMap, &kernel2d, stride, padding);
						}
						for (unsigned int wk = 0; wk < dw2d.size(); wk++) {
							for (unsigned int hk = 0; hk < dw2d.size(); hk++) {
								kernelGradient[d][wk][hk][dk] += dw2d[wk][hk];
							}
						}
					}
				}
			}
			for (unsigned int l = 0; l < kernel.size(); l++) {
				for (unsigned int i = 0; i < kernel[l].size(); i++) {
					for (unsigned int j = 0; j < kernel[l][i].size(); j++) {
						for (unsigned int k = 0; k < kernel[l][i][j].size(); k++) {
							kernelGradient[l][i][j][k] /= (float)input->size(); //batch Size
						}
					}
				}
			}
		});*/
		//d_input: conv w/ kernel = weights180, ActMap = dz
		gradient = vconvAL();
		//gradient.reserve(input->size());
		convA templatelp1 = convA(sl_1[0], std::vector<std::vector<float>>(sl_1[1], std::vector<float>(sl_1[2])));
		for (unsigned int i = 0; i < input->size(); i++) {
			//gradient.push_back(MathCNN::asStxxl(&(templatelp1)));
			gradient.push_back(templatelp1);
		}
		std::vector<std::thread> gradientThreads = std::vector<std::thread>(tDistribution.size());
		if (gradientThreads.size() == 0) {
			gradientThreads = std::vector<std::thread>(tDistribution.size());
		}
		for (unsigned int t = 0; t < tDistribution.size(); t++) {
			//gradientThreads[t] = thread([this, tDistribution, t, dz]() {
			gradientThreads[t] = std::thread([&size_, &sl_1_, &kernel_, &gradient_, input_, &padding_, &stride_, tDistribution, t, &dz_]() {
				for (unsigned int b = tDistribution[t][0]; b < tDistribution[t][1]; b++) {
					convA gradientT;
					for (unsigned int di = 0; di < sl_1_[2]; di++) { //depth input
						convA kernel_d(kernel_[0].size(), std::vector<std::vector<float>>(kernel_[0][0].size(), std::vector<float>(size_[2])));
						for (unsigned int d = 0; d < size_[2]; d++) { //depth
							for (unsigned int wk = 0; wk < kernel_[0].size(); wk++) {
								for (unsigned int hk = 0; hk < kernel_[0][wk].size(); hk++) {
									kernel_d[wk][hk][d] = kernel_[d][wk][hk][di];
								}
							}
						}
						convA kernel_d180 = MathCNN::rotate180(&kernel_d);
						convA* dz2 = const_cast<convA*>(&dz_[b]);
						//convAL* dz2 = const_cast<convAL*>(&dz_[b]);
						featureMap d_aMap = MathCNN::conv(dz2, &kernel_d180, stride_, (int)kernel_[0].size() - 1);
						gradientT.emplace_back(d_aMap);
					}
					for (unsigned int w = 0; w < sl_1_[0]; w++) {
						for (unsigned int h = 0; h < sl_1_[1]; h++) {
							for (unsigned int di = 0; di < sl_1_[2]; di++) {
								gradient_[b][w][h][di] = gradientT[di][w][h];
							}
						}
					}
				}
			});
		}
		for (unsigned int b = 0; b < tDistribution.size(); b++) {
			gradientThreads[b].join();
			kernelThreads[b].join();
		}
		//if(size[0] != 6) updateParameters();
		updateParameters();
	} catch (const std::system_error& e) {
		std::cout << "error " << e.code() << ": " << e.what() << std::endl;
		system("pause");
		std::cout << std::endl;
	}
};
void Conv::initialiseParameters(std::string actF) {
	float mean = (float)0;
	float dev = 1;
	if (actF == "ReLU") {
		dev = sqrt((float)2 / (float)(kernelS * kernelS * sl_1[2])); //He initialization
	}
	std::default_random_engine generator;
	for (unsigned int l = 0; l < size[2]; l++) {
		for (unsigned int i = 0; i < kernelS; i++) {
			for (unsigned int j = 0; j < kernelS; j++) {
				for (unsigned int k = 0; k < sl_1[2]; k++) {
					kernel[l][i][j][k] = MathCNN::getRandomNormal(&mean, &dev, &generator);
				}
			}
		}
	}
	kernelVel1.reserve(kernel.size());
	kernelVel2.reserve(kernel.size());
	for (unsigned int i = 0; i < kernel.size(); i++) {
		kernelVel1.emplace_back(convA(kernel[0].size(), std::vector<std::vector<float>>(kernel[0][0].size(), std::vector<float>(kernel[0][0][0].size()))));
		kernelVel2.emplace_back(convA(kernel[0].size(), std::vector<std::vector<float>>(kernel[0][0].size(), std::vector<float>(kernel[0][0][0].size()))));
	}
	avgM = std::vector<float>(size[2]);
	avgVar = std::vector<float>(size[2]);
	if (BN) {
		beta = std::vector<float>(size[2]);
		betaVel1 = std::vector<float>(size[2]);
		betaVel2 = std::vector<float>(size[2]);
		gamma.reserve(size[0]);
		gamma = std::vector<float>(size[2], 1);
		gammaVel1 = std::vector<float>(size[2]);
		gammaVel2 = std::vector<float>(size[2]);
		/*beta = convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2], 0)));
		betaVel1 = convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2], 0)));
		betaVel2 = convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2], 0)));
		gamma = convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2], 1)));
		gammaVel1 = convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2], 0)));
		gammaVel2 = convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2], 0)));*/
	}
	else {
		biases = std::vector<float>(size[2], 0);
		biasesVel1 = std::vector<float>(size[2], 0);
		biasesVel2 = std::vector<float>(size[2], 0);
	}
};
void Conv::setUpGradient(vconvAL* upGradient_) {
	upGradient = upGradient_;
};
void Conv::disableBN() {
	BN = false;
}
void Conv::endTraining() {
	converged = true;
	/*for (unsigned int i = 0; i < avgM.size(); i++) {
		avgM[i] /= (epoch-1);
	}
	float m = zws[0].size() * zws[0][0].size() * zws[0][0][0].size();
	for (unsigned int i = 0; i < avgM.size(); i++) {
		avgVar[i] *= m / ((m - 1) * (epoch - 1));
	}*/
	kernelGradient.clear();
	kernelGradient.shrink_to_fit();
	kernelVel1.clear();
	kernelVel1.shrink_to_fit();
	kernelVel2.clear();
	kernelVel2.shrink_to_fit();
	gammaGradient.clear();
	gammaGradient.shrink_to_fit();
	gammaVel1.clear();
	gammaVel1.shrink_to_fit();
	gammaVel2.clear();
	gammaVel2.shrink_to_fit();
	betaGradient.clear();
	betaGradient.shrink_to_fit();
	betaVel1.clear();
	betaVel1.shrink_to_fit();
	betaVel2.clear();
	betaVel2.shrink_to_fit();
	biasesGradient.clear();
	biasesGradient.shrink_to_fit();
	biasesVel1.clear();
	biasesVel1.shrink_to_fit();
	biasesVel2.clear();
	biasesVel2.shrink_to_fit();
	gradient.clear();
	//gradient.shrink_to_fit();
	activations.clear();
	//activations.shrink_to_fit();
};
void Conv::updateParameters() {
	if (BN) {
		//gamma, beta
	    /*for (unsigned int i = 0; i < beta.size(); i++) {
			for (unsigned int j = 0; j < beta[i].size(); j++) {
				for (unsigned int k = 0; k < beta[i][j].size(); k++) {
					betaVel1[i][j][k] = beta_o1 * betaVel1[i][j][k] + ((1 - beta_o1) * betaGradient[i][j][k]);
					betaVel2[i][j][k] = beta_o2 * betaVel2[i][j][k] + ((1 - beta_o2) * pow(betaGradient[i][j][k], 2));
					gammaVel1[i][j][k] = beta_o1 * gammaVel1[i][j][k] + ((1 - beta_o1) * gammaGradient[i][j][k]);
					gammaVel2[i][j][k] = beta_o2 * gammaVel2[i][j][k] + ((1 - beta_o2) * pow(gammaGradient[i][j][k], 2));
					float betaV1_h = betaVel1[i][j][k] / (1 - pow(beta_o1, epoch));
					float betaV2_h = betaVel2[i][j][k] / (1 - pow(beta_o2, epoch));
					float gammaV1_h = gammaVel1[i][j][k] / (1 - pow(beta_o1, epoch));
					float gammaV2_h = gammaVel2[i][j][k] / (1 - pow(beta_o2, epoch));
					beta[i][j][k] -= eta * (betaV1_h / sqrt(betaV2_h + 0.00001));
					gamma[i][j][k] -= eta * (gammaV1_h / sqrt(gammaV2_h + 0.00001));
				}
			}
		}*/
		for (unsigned int i = 0; i < size[2]; i++) {
			betaVel1[i] = beta_o1 * betaVel1[i] + ((1 - beta_o1) * betaGradient[i]);
			betaVel2[i] = beta_o2 * betaVel2[i] + ((1 - beta_o2) * pow(betaGradient[i], 2));
			gammaVel1[i] = beta_o1 * gammaVel1[i] + ((1 - beta_o1) * gammaGradient[i]);
			gammaVel2[i] = beta_o2 * gammaVel2[i] + ((1 - beta_o2) * pow(gammaGradient[i], 2));
			float betaV1_h = betaVel1[i] / (1 - pow(beta_o1, epoch));
			float betaV2_h = betaVel2[i] / (1 - pow(beta_o2, epoch));
			float gammaV1_h = gammaVel1[i] / (1 - pow(beta_o1, epoch));
			float gammaV2_h = gammaVel2[i] / (1 - pow(beta_o2, epoch));
			beta[i] -= eta * (betaV1_h / sqrt(betaV2_h + 0.00001));
			gamma[i] -= eta * (gammaV1_h / sqrt(gammaV2_h + 0.00001));
		}
	} else {
		//biases
		for (unsigned int i = 0; i < biases.size(); i++) {
			biasesVel1[i] = beta_o1 * biasesVel1[i] + ((1 - beta_o1) * biasesGradient[i]);
			biasesVel2[i] = beta_o2 * biasesVel2[i] + ((1 - beta_o2) * pow(biasesGradient[i], 2));
			float biasesV1_h = biasesVel1[i] / (1 - pow(beta_o1, epoch));
			float biasesV2_h = biasesVel2[i] / (1 - pow(beta_o2, epoch));
			biases[i] -= eta * (biasesV1_h / sqrt(biasesV2_h + 0.00001));
		}
	}
	//kernel
	for (unsigned int i = 0; i < kernel.size(); i++) {
		for (unsigned int j = 0; j < kernel[i].size(); j++) {
			for (unsigned int k = 0; k < kernel[i][j].size(); k++) {
				for (unsigned int l = 0; l < kernel[i][j][k].size(); l++) {
					kernelVel1[i][j][k][l] = beta_o1 * kernelVel1[i][j][k][l] + ((1 - beta_o1) * kernelGradient[i][j][k][l]);
					kernelVel2[i][j][k][l] = beta_o2 * kernelVel2[i][j][k][l] + ((1 - beta_o2) * pow(kernelGradient[i][j][k][l], 2));
					float kernelV1_h = kernelVel1[i][j][k][l] / (1 - pow(beta_o1, epoch));
					float kernelV2_h = kernelVel2[i][j][k][l] / (1 - pow(beta_o2, epoch));
					kernel[i][j][k][l] -= eta * (kernelV1_h / sqrt(kernelV2_h + 0.00001));
				}
			}
		}
	}
	epoch++;
};

std::vector<float> Conv::getAvgMean() {
	std::vector<float> avg = std::vector<float>(avgM);
	for (unsigned int i = 0; i < avgM.size(); i++) {
		avg[i] /= (epoch - 1);
	}
	return avg;
}

std::vector<float> Conv::getAvgVar() {
	std::vector<float> avg = std::vector<float>(avgVar);
	float m = zws[0].size() * zws[0][0].size() * zws[0][0][0].size();
	for (unsigned int i = 0; i < avgM.size(); i++) {
		avg[i] *= m / ((m - 1) * (epoch - 1));
	}
	return avg;
}