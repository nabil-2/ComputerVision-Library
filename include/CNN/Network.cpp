#include "Network.h"
#include "FileManager.h"
//#include "EasyBMP.hpp"
#include <typeinfo>
#include <fstream>
#include <iostream>
#include <regex>
#include <math.h>
#include <algorithm>
#include <filesystem>
#include <stxxl/vector>
#include "jExporter.h"

Network::Network() {
	std::ifstream inFile;
	//cout << "Dateipfad von hier. ";
	do {
		std::string filename;
		//cout << "Datei mit labels: ";
		//cin >> filename;
		//inFile.open(filename);
		inFile.open("labels.txt");
		if (!inFile) std::cout << "Datei existiert nicht. Erneut eingeben. "; std::cin;
	} while (!inFile);
	std::string fileText = "";
	while (!inFile.eof()) {
		std::string tmp;
		getline(inFile, tmp);
		fileText += tmp;
	}
	inFile.close();
	std::regex regex1("\\;");
	std::vector<std::string> split1(
		std::sregex_token_iterator(fileText.begin(), fileText.end(), regex1, -1),
		std::sregex_token_iterator()
	);
	for (unsigned int i = 0; i < split1.size(); i++) {
		std::string tmp = split1[i];
		std::regex regex2("\\,");
		std::vector<std::string> split2(
			std::sregex_token_iterator(tmp.begin(), tmp.end(), regex2, -1),
			std::sregex_token_iterator()
		);
		synsetID.push_back(std::vector<std::string>({split2[0], split2[1]}));
	}
};
Network::~Network() {
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			delete level[i][j];
		}
	}
};
int Network::addLevel() {
	level.push_back(std::vector<Layer*>());
	return level.size() - 1;
};
int Network::addLayer(std::string layerType, int size, int inputLevel, int inputLayer) {
	if (layerType == "Dense") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(Dense)) {
			Dense* inLayer = (dynamic_cast<Dense*>(layer));
			Dense* dense = new Dense(size, &(inLayer->activations));
			inLayer->setUpGradient(&(dense->gradient));
			level[level.size() - 1].push_back(dense);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(Flatten)) {
			Flatten* inLayer = (dynamic_cast<Flatten*>(layer));
			Dense* dense = new Dense(size, &(inLayer->activations));
			inLayer->setUpGradient(&(dense->gradient));
			level[level.size() - 1].push_back(dense);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(MultiDense)) {
			MultiDense* inLayer = (dynamic_cast<MultiDense*>(layer));
			Dense* dense = new Dense(size, &(inLayer->activations));
			inLayer->setUpGradient(&(dense->gradient));
			inLayer->nextW = false;
			level[level.size() - 1].push_back(dense);
			return level[level.size() - 1].size() - 1;
		}
	} else if (layerType == "MultiDense") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(MultiDense)) {
			MultiDense* inLayer = (dynamic_cast<MultiDense*>(layer));
			MultiDense* multiDense = new MultiDense(size, &(inLayer->activations), &(inLayer->position), &(inLayer->mean), inLayer->ixLayer + 1);
			inLayer->setUpGradient(&(multiDense->gradient));
			inLayer->wGp1 = &(multiDense->wG);
			inLayer->ap1 = &(multiDense->alphaM);
			inLayer->posp1 = &(multiDense->position);
			level[level.size() - 1].push_back(multiDense);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(MultiDenseInput)) {
			MultiDenseInput* inLayer = (dynamic_cast<MultiDenseInput*>(layer));
			float *mean = new float();
			*mean = 0;
			MultiDense* multiDense = new MultiDense(size, &(inLayer->activations), &(inLayer->position), mean, 1);
			inLayer->setUpGradient(&(multiDense->gradient));
			inLayer->wGp1 = &(multiDense->wG);
			inLayer->ap1 = &(multiDense->alphaM);
			inLayer->posp1 = &(multiDense->position);
			level[level.size() - 1].push_back(multiDense);
			return level[level.size() - 1].size() - 1;
		}
	} else if (layerType == "MultiDense2") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(MultiDense2)) {
			MultiDense2* inLayer = (dynamic_cast<MultiDense2*>(layer));
			MultiDense2* multiDense = new MultiDense2(size, &(inLayer->activations), &(inLayer->position), &(inLayer->mean), inLayer->ixLayer + 1);
			inLayer->setUpGradient(&(multiDense->gradient));
			inLayer->wGp1 = &(multiDense->wG);
			inLayer->ap1 = &(multiDense->alphaM);
			inLayer->posp1 = &(multiDense->position);
			level[level.size() - 1].push_back(multiDense);
			return level[level.size() - 1].size() - 1;
		}
		else if (typeid(*layer) == typeid(MultiDenseInput)) {
			MultiDenseInput* inLayer = (dynamic_cast<MultiDenseInput*>(layer));
			float* mean = new float();
			*mean = 0;
			MultiDense2* multiDense = new MultiDense2(size, &(inLayer->activations), &(inLayer->position), mean, 1);
			inLayer->setUpGradient(&(multiDense->gradient));
			inLayer->wGp1 = &(multiDense->wG);
			inLayer->ap1 = &(multiDense->alphaM);
			inLayer->posp1 = &(multiDense->position);
			inLayer->v2 = true;
			level[level.size() - 1].push_back(multiDense);
			return level[level.size() - 1].size() - 1;
		}
	}
	return -1;
};
int Network::addLayer(std::string layerType, int actMaps, int inputLevel, int inputLayer, int kernel, int stride, int padding) {
	if (layerType == "Conv") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(Conv)) {
			Conv* inLayer = (dynamic_cast<Conv*>(layer));
			Conv* conv = new Conv(actMaps, &(inLayer->activations), kernel, stride, padding);
			inLayer->setUpGradient(&(conv->gradient));
			level[level.size() - 1].push_back(conv);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(Input)) {
			Input* inLayer = (dynamic_cast<Input*>(layer));
			Conv* conv = new Conv(actMaps, &(inLayer->activations), kernel, stride, padding);
			level[level.size() - 1].push_back(conv);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(Pooling)) {
			Pooling* inLayer = (dynamic_cast<Pooling*>(layer));
			Conv* conv = new Conv(actMaps, &(inLayer->activations), kernel, stride, padding);
			inLayer->setUpGradient(&(conv->gradient));
			level[level.size() - 1].push_back(conv);
			return level[level.size() - 1].size() - 1;
		}
	}
	return -1;
};
int Network::addLayer(int kernel, std::string layerType, int inputLevel, int inputLayer) {
	if (layerType == "Pooling") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(Conv)) {
			Conv* inLayer = (dynamic_cast<Conv*>(layer));
			Pooling* pooling = new Pooling(&(inLayer->activations), kernel);
			inLayer->setUpGradient(&(pooling->gradient));
			inLayer->disableBN();
			level[level.size() - 1].push_back(pooling);
			return level[level.size() - 1].size() - 1;
		}
	}
	return -1;
};
int Network::addLayer(std::string layerType, int inputLevel, int inputLayer) {
	if (layerType == "Flatten") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(Conv)) {
			Conv* inLayer = (dynamic_cast<Conv*>(layer));
			Flatten* flatten = new Flatten(&(inLayer->activations));
			inLayer->setUpGradient(&(flatten->gradient));
			level[level.size() - 1].push_back(flatten);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(Input)) {
			Input* inLayer = (dynamic_cast<Input*>(layer));
			Flatten* flatten = new Flatten(&(inLayer->activations));
			level[level.size() - 1].push_back(flatten);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(Pooling)) {
			Pooling* inLayer = (dynamic_cast<Pooling*>(layer));
			Flatten* flatten = new Flatten(&(inLayer->activations));
			inLayer->setUpGradient(&(flatten->gradient));
			level[level.size() - 1].push_back(flatten);
			return level[level.size() - 1].size() - 1;
		}
	} else if (layerType == "Softmax") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(Dense)) {
			Dense* inLayer = (dynamic_cast<Dense*>(layer));
			Softmax* softmax = new Softmax(&(inLayer->activations));
			inLayer->setUpGradient(&(softmax->gradient));
			inLayer->lastLayer = true;
			level[level.size() - 1].push_back(softmax);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(Flatten)) {
			Flatten* inLayer = (dynamic_cast<Flatten*>(layer));
			Softmax* softmax = new Softmax(&(inLayer->activations));
			inLayer->setUpGradient(&(softmax->gradient));
			level[level.size() - 1].push_back(softmax);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(MultiDense)) {
			MultiDense* inLayer = (dynamic_cast<MultiDense*>(layer));
			Softmax* softmax = new Softmax(&(inLayer->activations));
			inLayer->setUpGradient(&(softmax->gradient));
			inLayer->lastLayer = true;
			level[level.size() - 1].push_back(softmax);
			return level[level.size() - 1].size() - 1;
		} else if (typeid(*layer) == typeid(MultiDense2)) {
			MultiDense2* inLayer = (dynamic_cast<MultiDense2*>(layer));
			Softmax* softmax = new Softmax(&(inLayer->activations));
			inLayer->setUpGradient(&(softmax->gradient));
			inLayer->lastLayer = true;
			level[level.size() - 1].push_back(softmax);
			return level[level.size() - 1].size() - 1;
		}
	} else if (layerType == "MultiDenseInput") {
		Layer* layer = level[inputLevel][inputLayer];
		if (typeid(*layer) == typeid(Dense)) {
			Dense* inLayer = (dynamic_cast<Dense*>(layer));
			MultiDenseInput* multiDenseInput = new MultiDenseInput(&(inLayer->activations));
			inLayer->setUpGradient(&(multiDenseInput->gradient));
			level[level.size() - 1].push_back(multiDenseInput);
			return level[level.size() - 1].size() - 1;
		}
		else if (typeid(*layer) == typeid(Flatten)) {
			Flatten* inLayer = (dynamic_cast<Flatten*>(layer));
			MultiDenseInput* multiDenseInput = new MultiDenseInput(&(inLayer->activations));
			inLayer->setUpGradient(&(multiDenseInput->gradient));
			level[level.size() - 1].push_back(multiDenseInput);
			return level[level.size() - 1].size() - 1;
		}
	}
	return -1;
};
int Network::addLayer(std::string layer, int* size) {
	if (layer == "Input" && level.size() == 1) {
		Input* input = new Input(size);
		level[level.size() - 1].push_back(input);
		return level[level.size() - 1].size() - 1;
	}
	return -1;
};
void Network::intialiseParameters() {
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			level[i][j]->initialiseParameters(actF);
		}
	}
}
void Network::setActivationFunction(std::string actFunction) {
	actF = actFunction;
};
/*void Network::setInput(std::vector<convA> *in) {
	vconvAL in_stxxl = MathCNN::asStxxl(in);
	setInput(&in_stxxl);
};*/
void Network::setInput(vconvAL* in) {
	for (unsigned int i = 0; i < (*in).size(); i++) {
		(*in)[i] = preprocess(&((*in)[i]));
	}
	for (unsigned int i = 0; i < (*in).size(); i++) {
		for (unsigned int j = 0; j < (*in)[i].size(); j++) {
			for (unsigned int k = 0; k < (*in)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*in)[i][j][k].size(); l++) {
					(*in)[i][j][k][l] -= batchMean;
				}
			}
		}
	}
	Input* input = dynamic_cast<Input*>(level[0][0]);
	input->activations = *in;
};
/*void Network::setInput(convA in) {
	Input* input = dynamic_cast<Input*>(level[0][0]);
	input->activations = vector<convA>(1, in);
};*/
std::vector<std::vector<std::string>> Network::predict() {
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			//std::cout << "forward " << i << std::endl;
			//system("pause");
			level[i][j]->forward();
		}
	}
	Layer* lastLayer = level[level.size() - 1][0];
	if (typeid(*lastLayer) == typeid(Softmax)) {
		return encodeOutput(dynamic_cast<Softmax*>(lastLayer)->activations);
	} else if (typeid(*lastLayer) == typeid(Dense)) {
		return encodeOutput(dynamic_cast<Dense*>(lastLayer)->activations);
	} else if (typeid(*lastLayer) == typeid(MultiDense)) {
		return encodeOutput(dynamic_cast<MultiDense*>(lastLayer)->activations);
	}
	return std::vector<std::vector<std::string>>(0);
};
std::vector<denseA> Network::predictVal() {
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			//std::cout << "forward " << i << std::endl;
			//system("pause");
			level[i][j]->forward();
		}
	}
	Layer* lastLayer = level[level.size() - 1][0];
	if (typeid(*lastLayer) == typeid(Softmax)) {
		return dynamic_cast<Softmax*>(lastLayer)->activations;
	}
	else if (typeid(*lastLayer) == typeid(Dense)) {
		return dynamic_cast<Dense*>(lastLayer)->activations;
	}
	else if (typeid(*lastLayer) == typeid(MultiDense)) {
		return dynamic_cast<MultiDense*>(lastLayer)->activations;
	}
	return std::vector<denseA>(0);
};
convAL Network::preprocess(convAL* img) {
	Input* inputL = dynamic_cast<Input*>(level[0][0]);
	int widthD = inputL->activations[0].size(),
		heightD = inputL->activations[0][0].size();
	int width = img[0].size(),
		height = img[0][0].size();
	//convA newImg(widthD, std::vector<std::vector<float>>(heightD));
	convAL newImg;
	float scaleX = (float)width / (float)widthD;
	float scaleY = (float)height / (float)heightD;
	//EasyBMP::Image imgage(224, 224, "sample.bmp");
	for (int x = 0; x < widthD; x++) {
		newImg.push_back(vectorff());
		for (int y = 0; y < heightD; y++) {
			newImg[x].push_back(vectorf());
			int pX = floor(scaleX * (float)x);
			int pY = floor(scaleY * (float)y);
			newImg[x][y] = (*img)[pX][pY];
			//imgage.SetPixel(x, y, EasyBMP::RGBColor(newImg[x][y][0]*255, newImg[x][y][1] * 255, newImg[x][y][2] * 255));
		}
	}
	//imgage.Write();
	return newImg;
};
void Network::epoch(std::vector<std::vector<std::string>> *data) { //data = [[path, label], [...], ...];
	//cout << (*data)[0][0] << endl;
	//if ((*data)[0][1] == "9") return;
	//std::vector<convA> imgs;
	//system("pause");
	vconvAL imgs;
	for (unsigned int i = 0; i < data->size(); i++) {
		//cout << "reading img " << i << endl;
		convA img = FileManager::getInput((*data)[i][0]);
		//imgs.push_back(MathCNN::asStxxl(&img));
		//img = preprocess(&img);
		imgs.push_back(img);
	}
	setInput(&imgs);
	std::vector<denseA> result = predictVal();
	//system("pause");
	/*for (unsigned int i = 0; i < result[0].size(); i++) {
		//cout << result[0][i] << endl;
	}*/
	//system("pause");
	std::vector<denseA> resultD;
	for (unsigned int i = 0; i < result.size(); i++) {
		resultD.push_back(denseA());
		for (unsigned int j = 0; j < result[i].size(); j++) {
			/*if ((*data)[i][1] == synsetID[j][0]) {
				resultD[i].push_back(1);
			} else {
				resultD[i].push_back(0);
			}*/
			if (j == stoi((*data)[i][1])) {
				resultD[i].push_back(1);
			} else {
				resultD[i].push_back(0);
			}
		}
	}
	/*cout << "resultD" << endl;
	for (unsigned int i = 0; i < resultD[0].size(); i++) {
		cout << resultD[0][i] << endl;
	}*/
	//system("pause");
	std::vector<denseA> dCa;
	if (costF == "cross-entropy") {
		float avgError = 0;
		for (unsigned int i = 0; i < result.size(); i++) {
			float sum = 0;
			for (unsigned int j = 0; j < result[i].size(); j++) {
				sum += resultD[i][j] * log(result[i][j]);
			}
			sum *= -1;
			avgError += sum;
		}
		avgError /= result.size();
		for (unsigned int i = 0; i < result.size(); i++) {
			dCa.push_back(denseA());
			for (unsigned int j = 0; j < result[i].size(); j++) {
				dCa[i].push_back((-1)*resultD[i][j] / result[i][j]);
			}
		}
		std::cout << "error: " + std::to_string(avgError) << ", ";
		//avgE += (*data)[0][1] + "_" + to_string(avgError) + ";";
		avgE += std::to_string(avgError) + ";";
	} else if(costF == "mean-squared") {}
	std::ofstream outfile("plotData.txt");
	outfile << avgE;
	outfile.close();
	Softmax* softmax = dynamic_cast<Softmax*>(level[level.size() - 1][0]);
	softmax->setUpGradient(&dCa);
	softmax->resultD = resultD;
	//system("pause");
	std::cout << "backward 00";
	for (unsigned int i = level.size() - 1; i >= 1; i--) {
		std::cout << std::string(std::to_string(i+1).length(), '\b') << std::string((std::to_string(i + 1).length() > 1 && std::to_string(i).length() == 1) ? 1 : 0, ' ') << i;
		//system("pause");
		level[i][0]->backward();
	}
	std::cout << std::endl;
	//system("pause");
	std::string pos;
	bool plot2 = false;
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			bool plot = false;
			std::vector<std::vector<float>>* positions = nullptr;
			if (typeid(*level[i][j]) == typeid(MultiDense)) {
				MultiDense* layer = dynamic_cast<MultiDense*>(level[i][j]);
				positions = &(layer->position);
				plot = true;
			} else if (typeid(*level[i][j]) == typeid(MultiDenseInput)) {
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>(level[i][j]);
				positions = &(layer->position);
				plot = true;
			} else if (typeid(*level[i][j]) == typeid(MultiDense2)) {
				MultiDense2* layer = dynamic_cast<MultiDense2*>(level[i][j]);
				positions = &(layer->position);
				plot = true;
			}
			if (plot) {
				for (unsigned int k = 0; k < positions->size(); k++) {
					std::string tmp = "";
					for (unsigned int l = 0; l < (*positions)[k].size(); l++) {
						tmp += std::to_string((*positions)[k][l]);
						tmp += ',';
					}
					pos += tmp + ";";
				}
				plot2 = true;
			}
		}
	}
	if (plot2) {
		std::ofstream outfile2("posData.txt");
		outfile2 << pos;
		outfile2.close();
	}
	/**/
};
void Network::setCostFunction(std::string costFct) {
	if (costF == "cross-entropy" || costF == "mean-squared") costF = costFct;
};
std::vector<std::vector<std::string>> Network::encodeOutput(std::vector<denseA> output) {
	std::vector<std::vector<std::string>> result;
	for (unsigned int i = 0; i < output.size(); i++) {
		std::vector<std::string> values;
		for (unsigned int j = 0; j < output[i].size(); j++) {
			float val = output[i][j];
			float percent = roundf(val * 10000) / 100;
			std::string st = "";
			if (percent < 10) st += "0";
			st += std::to_string(percent);
			std::string name = st.substr(0, st.size() - 4) + "% " + synsetID[j][1];
			values.push_back(name);
		}
		sort(values.begin(), values.end());
		reverse(values.begin(), values.end());
		result.push_back(values);
	}
	return result;
};
std::vector<denseA> Network::sortOutput(std::vector<denseA> output) {
	std::vector<denseA> ret = output;
	for (unsigned int i = 0; i < ret.size(); i++) {
		std::sort(ret[i].begin(), ret[i].end(), std::greater<float>());
	}
	return ret;
};
void Network::finishTraining() {
	Input* in = dynamic_cast<Input*>(level[0][0]);
	in->activations.clear();
	//in->activations.shrink_to_fit();
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			if (typeid(*level[i][j]) == typeid(Dense)) {
				Dense* layer = dynamic_cast<Dense*>(level[i][j]);
				layer->endTraining();
			} else if (typeid(*level[i][j]) == typeid(MultiDense)) {
				MultiDense* layer = dynamic_cast<MultiDense*>(level[i][j]);
				layer->endTraining();
			} else if (typeid(*level[i][j]) == typeid(MultiDenseInput)) {
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>(level[i][j]);
				layer->endTraining();
			} else if (typeid(*level[i][j]) == typeid(Conv)) {
				Conv* layer = dynamic_cast<Conv*>(level[i][j]);
				layer->endTraining();
			} else if (typeid(*level[i][j]) == typeid(Flatten)) {
				Flatten* layer = dynamic_cast<Flatten*>(level[i][j]);
				layer->endTraining();
			}
		}
	}
};
void Network::exportNet(std::string filename) {
	std::filesystem::remove(filename);
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			if (typeid(*level[i][j]) == typeid(Dense)) {
				//std::cout << "export dense level " << i << std::endl;
				//system("pause");
				Dense* layer = dynamic_cast<Dense*>(level[i][j]);
				denseExp data;
				data.weights = &layer->weights;
				data.beta = &layer->beta;
				data.gamma = &layer->gamma;
				denseA avgM = layer->getAvgMean();
				denseA avgVar = layer->getAvgVar();
				data.avgM = &avgM;
				data.avgVar = &avgVar;
				jExporter::exportData(filename, data);

				//std::cout << (*data.avgM)[0] << std::endl;
				//system("pause");
			}
			else if (typeid(*level[i][j]) == typeid(Conv)) {
				Conv* layer = dynamic_cast<Conv*>(level[i][j]);
				convExp data;
				data.kernel = &layer->kernel;
				data.beta = &layer->beta;
				data.gamma = &layer->gamma;
				data.biases = &layer->biases;
				denseA avgM = layer->getAvgMean();
				denseA avgVar = layer->getAvgVar();
				data.avgM = &avgM;
				data.avgVar = &avgVar;
				jExporter::exportData(filename, data);
			}
			else if (typeid(*level[i][j]) == typeid(Input)) {
				Input* layer = dynamic_cast<Input*>(level[i][j]);
				inputExp data;
				data.batchMean = &batchMean;
				jExporter::exportData(filename, data);
			}
			else if (typeid(*level[i][j]) == typeid(MultiDense)) {
				MultiDense* layer = dynamic_cast<MultiDense*>(level[i][j]);
				multiDenseExp data;
				data.position = &layer->position;
				data.avgM = &layer->avgM;
				data.avgVar = &layer->avgVar;
				data.betaM = &layer->betaM;
				data.alphaM = &layer->alphaM;
				data.beta = &layer->beta;
				data.gamma = &layer->gamma;
				jExporter::exportData(filename, data);
			}
			else if (typeid(*level[i][j]) == typeid(MultiDense2)) {
				MultiDense2* layer = dynamic_cast<MultiDense2*>(level[i][j]);
				multiDenseExp data;
				data.position = &layer->position;
				data.avgM = &layer->avgM;
				data.avgVar = &layer->avgVar;
				data.betaM = &layer->betaM;
				data.alphaM = &layer->alphaM;
				data.beta = &layer->beta;
				data.gamma = &layer->gamma;
				jExporter::exportData(filename, data);
			}
			else if (typeid(*level[i][j]) == typeid(MultiDenseInput)) {
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>(level[i][j]);
				multiDenseInputExp data;
				data.position = &layer->position;
				jExporter::exportData(filename, data);
			}
			else {
				std::ofstream out;
				out.open(filename, std::ios_base::app);
				out << "{}\n";
				out.close();
			}
		}
	}
	//FileManager::exportNet(filename, batchMean, &level);
	/*std::filesystem::remove(filename);
	std::ofstream out;
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			//cout << string(to_string(i + 1).length() + 3, '\b') << string((to_string(i + 1).length() > 1 && to_string(i).length() == 1) ? 1 : 0, ' ') << i << "/" << level.size();
			//cout << "writing level " << i << endl;
			out.open(filename, std::ios_base::app);
			json* layerJ = FileManager::getJSON();//json({});
			if (typeid(*level[i][j]) == typeid(Dense)) {
				Dense* layer = dynamic_cast<Dense*>(level[i][j]);
				layerJ["weights"] = json(layer->weights);
				layerJ["beta"] = json(layer->beta);
				layerJ["gamma"] = json(layer->gamma);
				layerJ["mean"] = json(layer->getAvgMean()); //get Avg
				layerJ["variance"] = json(layer->getAvgVar()); //get Avg
			} else if (typeid(*level[i][j]) == typeid(Conv)) {
				Conv* layer = dynamic_cast<Conv*>(level[i][j]);
				layerJ["kernel"] = json(layer->kernel);
				layerJ["beta"] = json(layer->beta);
				layerJ["gamma"] = json(layer->gamma);
				layerJ["bias"] = json(layer->biases);
				layerJ["mean"] = json(layer->getAvgMean()); //get Avg
				layerJ["variance"] = json(layer->getAvgVar()); //get Avg
				//cout << "variance level " << i << ": " << layer->avgVar[0] << endl;
			} else if (typeid(*level[i][j]) == typeid(Input)) {
				Input* layer = dynamic_cast<Input*>(level[i][j]);
				layerJ["batchMean"] = batchMean;
			} else if (typeid(*level[i][j]) == typeid(MultiDense)) {
				MultiDense* layer = dynamic_cast<MultiDense*>(level[i][j]);
				layerJ["avgM"] = json(layer->avgM);
				layerJ["avgVar"] = json(layer->avgVar);
				layerJ["betaM"] = json(layer->betaM);
				layerJ["alphaM"] = json(layer->alphaM);
				layerJ["beta"] = json(layer->beta);
				layerJ["gamma"] = json(layer->gamma);
				layerJ["position"] = json(layer->position);
			} else if (typeid(*level[i][j]) == typeid(MultiDenseInput)) {
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>(level[i][j]);
				layerJ["position"] = json(layer->position);
			}
			//string layer = layerJ.dump() + "\n";
			//out << layer;
			out << *layerJ;
			out.close();
			out.open(filename, std::ios_base::app);
			out << "\n";
			out.close();
		}
	}*/
};
void Network::importNet(std::string filename) {
	//std::cout << "reading..." << std::endl;
	std::vector<std::thread> readThreads;
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			//cout << string(to_string(i + 1).length() + 3, '\b') << string((to_string(i + 1).length() > 1 && to_string(i).length() == 1) ? 1 : 0, ' ') << i << "/" << level.size();
			
			if (typeid(*level[i][j]) == typeid(Dense)) {
				Dense* layer = dynamic_cast<Dense*>(level[i][j]);
				readThreads.push_back(std::thread([layer, i, filename] {
					jExporter::importData(i, filename, &(layer->weights), &(layer->beta), &(layer->gamma), &(layer->avgM), &(layer->avgVar));
				}));
				layer->converged = true;
			}
			else if (typeid(*level[i][j]) == typeid(Conv)) {
				Conv* layer = dynamic_cast<Conv*>(level[i][j]);
				readThreads.push_back(std::thread([layer, i, filename] {
					jExporter::importData(i, filename, &(layer->kernel), &(layer->beta), &(layer->gamma), &(layer->biases), &(layer->avgM), &(layer->avgVar));
				}));
				layer->converged = true;
			}
			else if (typeid(*level[i][j]) == typeid(Input)) {
				Input* layer = dynamic_cast<Input*>(level[i][j]);
				jExporter::importData(i, filename, &batchMean);
			}
			else if (typeid(*level[i][j]) == typeid(MultiDense)) {
				MultiDense* layer = dynamic_cast<MultiDense*>(level[i][j]);
				jExporter::importData(i, filename,
					&(layer->avgM),
					&(layer->avgVar),
					&(layer->betaM),
					&(layer->alphaM),
					&(layer->beta),
					&(layer->gamma),
					&(layer->position)
				);
				layer->converged = true;
			}
			else if (typeid(*level[i][j]) == typeid(MultiDense2)) {
				MultiDense2* layer = dynamic_cast<MultiDense2*>(level[i][j]);
				jExporter::importData(i, filename,
					&(layer->avgM),
					&(layer->avgVar),
					&(layer->betaM),
					&(layer->alphaM),
					&(layer->beta),
					&(layer->gamma),
					&(layer->position)
				);
				layer->converged = true;
			}
			else if (typeid(*level[i][j]) == typeid(MultiDenseInput)) {
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>(level[i][j]);
				jExporter::importData(i, filename, &(layer->position));
			}
		}
	}
	for (unsigned int i = 0; i < readThreads.size(); i++) {
		readThreads[i].join();
	}
	//FileManager::importNet(filename, &batchMean, &level);
	/*std::ifstream inFile;
	std::cout << "reading..." << std::endl;
	for (unsigned int i = 0; i < level.size(); i++) {
		for (unsigned int j = 0; j < level[i].size(); j++) {
			//cout << string(to_string(i + 1).length() + 3, '\b') << string((to_string(i + 1).length() > 1 && to_string(i).length() == 1) ? 1 : 0, ' ') << i << "/" << level.size();
			//cout << "reading level " << i << endl;
			if (typeid(*level[i][j]) == typeid(Dense)) {
				inFile.open(filename);
				std::string data;
				int line = 1;
				while (!inFile.eof()) {
					std::string tmp;
					getline(inFile, tmp);
					if (line == i+1) {
						data = tmp;
						break;
					}
					line++;
				}
				json layerJ = json::parse(data);
				Dense* layer = dynamic_cast<Dense*>(level[i][j]);
				layer->weights = layerJ["weights"].get<denseW>();
				layer->beta = layerJ["beta"].get<denseA>();
				layer->gamma = layerJ["gamma"].get<denseA>();
				layer->avgM = layerJ["mean"].get<denseA>();
				layer->avgVar = layerJ["variance"].get<denseA>();
				layer->converged = true;
				inFile.close();
			} else if (typeid(*level[i][j]) == typeid(Conv)) {
				inFile.open(filename);
				std::string data;
				int line = 1;
				while (!inFile.eof()) {
					std::string tmp;
					getline(inFile, tmp);
					if (line == i+1) {
						data = tmp;
						break;
					}
					line++;
				}
				json layerJ = json::parse(data);
				Conv* layer = dynamic_cast<Conv*>(level[i][j]);
				layer->kernel = layerJ["kernel"].get<std::vector<convA>>();
				layer->beta = layerJ["beta"].get<std::vector<float>>();
				layer->gamma = layerJ["gamma"].get<std::vector<float>>();
				layer->biases = layerJ["bias"].get<std::vector<float>>();
				layer->avgM = layerJ["mean"].get<std::vector<float>>();
				layer->avgVar = layerJ["variance"].get<std::vector<float>>();
				layer->converged = true;
				inFile.close();
			} else if (typeid(*level[i][j]) == typeid(Input)) {
				inFile.open(filename);
				std::string data;
				int line = 1;
				while (!inFile.eof()) {
					std::string tmp;
					getline(inFile, tmp);
					if (line == i + 1) {
						data = tmp;
						break;
					}
					line++;
				}
				json layerJ = json::parse(data);
				Input* layer = dynamic_cast<Input*>(level[i][j]);
				batchMean = layerJ["batchMean"].get<float>();
				inFile.close();
			} else if (typeid(*level[i][j]) == typeid(MultiDense)) {
				inFile.open(filename);
				std::string data;
				int line = 1;
				while (!inFile.eof()) {
					std::string tmp;
					getline(inFile, tmp);
					if (line == i + 1) {
						data = tmp;
						break;
					}
					line++;
				}
				json layerJ = json::parse(data);
				MultiDense* layer = dynamic_cast<MultiDense*>(level[i][j]);
				layer->avgM = layerJ["avgM"].get<float>();
				layer->avgVar = layerJ["avgVar"].get<float>();
				layer->betaM = layerJ["betaM"].get<denseA>();
				layer->alphaM = layerJ["alphaM"].get<denseA>();
				layer->beta = layerJ["beta"].get<denseA>();
				layer->gamma = layerJ["gamma"].get<denseA>();
				layer->position = layerJ["position"].get<std::vector<std::vector<float>>>();
				inFile.close();
			} else if (typeid(*level[i][j]) == typeid(MultiDenseInput)) {
				inFile.open(filename);
				std::string data;
				int line = 1;
				while (!inFile.eof()) {
					std::string tmp;
					getline(inFile, tmp);
					if (line == i + 1) {
						data = tmp;
						break;
					}
					line++;
				}
				json layerJ = json::parse(data);
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>(level[i][j]);
				layer->position = layerJ["position"].get<std::vector<std::vector<float>>>();
				inFile.close();
			}
		}
	}*/
};
void Network::calcBatchMean(int size, std::string dir) {
	int i = 0;
	for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
		if (!std::filesystem::is_directory(entry.path())) {
			std::string path{ entry.path().u8string() };
			convA img = FileManager::getInput(path);
			//convAL img = MathCNN::asStxxl(&img0);
			img = preprocess(&img);
			for (unsigned int j = 0; j < img.size(); j++) {
				for (unsigned int k = 0; k < img[j].size(); k++) {
					for (unsigned int l = 0; l < img[j][k].size(); l++) {
						batchMean += img[j][k][l];
					}
				}
			}
		}
		i++;
		if (i == size) break;
	}
	Input* inputL = dynamic_cast<Input*>(level[0][0]);
	int width = inputL->activations[0].size(),
		height = inputL->activations[0][0].size(),
		channels = inputL->activations[0][0][0].size();
	batchMean /= size * width * height * channels;
};