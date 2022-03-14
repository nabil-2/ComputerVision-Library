#include "jExporter.h"

#include "nlohmann.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

typedef std::vector<float> denseA;
typedef std::vector<std::vector<std::vector<float>>> convA;
typedef std::vector<std::vector<float>> denseW;
typedef std::vector<std::vector<float>> featureMap;

void jExporter::exportData(std::string filename, denseExp layer) {
	std::ofstream out;
	out.open(filename, std::ios_base::app);
	json layerJ = json({});
	layerJ["weights"] = json(*layer.weights);
	layerJ["beta"] = json(*layer.beta);
	layerJ["gamma"] = json(*layer.gamma);
	layerJ["mean"] = json(*layer.avgM);
	layerJ["variance"] = json(*layer.avgVar);
	out << layerJ;
	out.close();
	out.open(filename, std::ios_base::app);
	out << "\n";
	out.close();
};
void jExporter::exportData(std::string filename, convExp layer) {
	std::ofstream out;
	out.open(filename, std::ios_base::app);
	json layerJ = json({});
	layerJ["kernel"] = json(*(layer.kernel));
	layerJ["beta"] = json(*layer.beta);
	layerJ["gamma"] = json(*layer.gamma);
	layerJ["bias"] = json(*layer.biases);
	layerJ["mean"] = json(*layer.avgM);
	layerJ["variance"] = json(*layer.avgVar);
	out << layerJ;
	out.close();
	out.open(filename, std::ios_base::app);
	out << "\n";
	out.close();
};
void jExporter::exportData(std::string filename, inputExp layer) {
	std::ofstream out;
	out.open(filename, std::ios_base::app);
	json layerJ = json({});
	layerJ["batchMean"] = json(*layer.batchMean);
	out << layerJ;
	out.close();
	out.open(filename, std::ios_base::app);
	out << "\n";
	out.close();
};
void jExporter::exportData(std::string filename, multiDenseExp layer) {
	std::ofstream out;
	out.open(filename, std::ios_base::app);
	json layerJ = json({});
	layerJ["avgM"] = json(*layer.avgM);
	layerJ["avgVar"] = json(*layer.avgVar);
	layerJ["betaM"] = json(*layer.betaM);
	layerJ["alphaM"] = json(*(layer.alphaM));
	layerJ["beta"] = json(*layer.beta);
	layerJ["gamma"] = json(*layer.gamma);
	layerJ["position"] = json(*layer.position);
	out << layerJ;
	out.close();
	out.open(filename, std::ios_base::app);
	out << "\n";
	out.close();
};
void jExporter::exportData(std::string filename, multiDenseInputExp layer) {
	std::ofstream out;
	out.open(filename, std::ios_base::app);
	json layerJ = json({});
	layerJ["position"] = json(*layer.position);
	out << layerJ;
	out.close();
	out.open(filename, std::ios_base::app);
	out << "\n";
	out.close();
};


void jExporter::importData(int i, std::string filename, denseW* weights, denseA* beta, denseA* gamma, denseA* avgM, denseA* avgVar) {
	std::ifstream inFile;
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
	inFile.close();
	json layerJ = json::parse(data);
	*weights = layerJ["weights"].get<denseW>();
	*beta = layerJ["beta"].get<denseA>();
	*gamma = layerJ["gamma"].get<denseA>();
	*avgM = layerJ["mean"].get<denseA>();
	*avgVar = layerJ["variance"].get<denseA>();
}; //dense
void jExporter::importData(int i, std::string filename, std::vector<j::convA>* kernel, j::denseA* beta, denseA* gamma, j::denseA* biases, j::denseA* avgM, j::denseA* avgVar) {
	std::ifstream inFile;
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
	inFile.close();
	json layerJ = json::parse(data);
	*kernel = layerJ["kernel"].get<std::vector<convA>>();
	*beta = layerJ["beta"].get<std::vector<float>>();
	*gamma = layerJ["gamma"].get<std::vector<float>>();
	*biases = layerJ["bias"].get<std::vector<float>>();
	*avgM = layerJ["mean"].get<std::vector<float>>();
	*avgVar = layerJ["variance"].get<std::vector<float>>();

}; //conv
void jExporter::importData(int i, std::string filename, float* batchMean) {
	std::ifstream inFile;
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
	inFile.close();
	json layerJ = json::parse(data);
	*batchMean = layerJ["batchMean"].get<float>();
}; //input
void jExporter::importData(int i, std::string filename, denseA* avgM, denseA* avgVar, denseA* betaM, denseA* alphaM, denseA* beta, denseA* gamma, denseW* position) {
	std::ifstream inFile;
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
	inFile.close();
	json layerJ = json::parse(data);
	*avgM = layerJ["avgM"].get<denseA>();
	*avgVar = layerJ["avgVar"].get<denseA>();
	*betaM = layerJ["betaM"].get<denseA>();
	*alphaM = layerJ["alphaM"].get<denseA>();
	*beta = layerJ["beta"].get<denseA>();
	*gamma = layerJ["gamma"].get<denseA>();
	*position = layerJ["position"].get<std::vector<std::vector<float>>>();

}; //multiDense
void jExporter::importData(int i, std::string filename, denseW* position) {
	std::ifstream inFile;
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
	inFile.close();
	json layerJ = json::parse(data);
}; //multiDenseInput


/*
void jExporter::exportNet(std::string filename, float batchMean, std::vector<std::vector<Layer*>>* level) {
	std::filesystem::remove(filename);
	std::ofstream out;
	for (unsigned int i = 0; i < level->size(); i++) {
		for (unsigned int j = 0; j < (*level)[i].size(); j++) {
			//cout << string(to_string(i + 1).length() + 3, '\b') << string((to_string(i + 1).length() > 1 && to_string(i).length() == 1) ? 1 : 0, ' ') << i << "/" << level.size();
			//cout << "writing level " << i << endl;
			out.open(filename, std::ios_base::app);
			json layerJ = json({});
			if (typeid(*(*level)[i][j]) == typeid(Dense)) {
				Dense* layer = dynamic_cast<Dense*>((*level)[i][j]);
				layerJ["weights"] = json(layer->weights);
				layerJ["beta"] = json(layer->beta);
				layerJ["gamma"] = json(layer->gamma);
				layerJ["mean"] = json(layer->getAvgMean()); //get Avg
				layerJ["variance"] = json(layer->getAvgVar()); //get Avg
			}
			else if (typeid(*(*level)[i][j]) == typeid(Conv)) {
				Conv* layer = dynamic_cast<Conv*>((*level)[i][j]);
				layerJ["kernel"] = json(layer->kernel);
				layerJ["beta"] = json(layer->beta);
				layerJ["gamma"] = json(layer->gamma);
				layerJ["bias"] = json(layer->biases);
				layerJ["mean"] = json(layer->getAvgMean()); //get Avg
				layerJ["variance"] = json(layer->getAvgVar()); //get Avg
				//cout << "variance level " << i << ": " << layer->avgVar[0] << endl;
			}
			else if (typeid(*(*level)[i][j]) == typeid(Input)) {
				Input* layer = dynamic_cast<Input*>((*level)[i][j]);
				layerJ["batchMean"] = batchMean;
			}
			else if (typeid(*(*level)[i][j]) == typeid(MultiDense)) {
				MultiDense* layer = dynamic_cast<MultiDense*>((*level)[i][j]);
				layerJ["avgM"] = json(layer->avgM);
				layerJ["avgVar"] = json(layer->avgVar);
				layerJ["betaM"] = json(layer->betaM);
				layerJ["alphaM"] = json(layer->alphaM);
				layerJ["beta"] = json(layer->beta);
				layerJ["gamma"] = json(layer->gamma);
				layerJ["position"] = json(layer->position);
			}
			else if (typeid(*(*level)[i][j]) == typeid(MultiDenseInput)) {
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>((*level)[i][j]);
				layerJ["position"] = json(layer->position);
			}
			//string layer = layerJ.dump() + "\n";
			//out << layer;
			out << layerJ;
			out.close();
			out.open(filename, std::ios_base::app);
			out << "\n";
			out.close();
		}
	}
};
void FileManager::importNet(std::string filename, float *batchMean, std::vector<std::vector<Layer*>>* level) {
	std::ifstream inFile;
	std::cout << "reading..." << std::endl;
	for (unsigned int i = 0; i < (*level).size(); i++) {
		for (unsigned int j = 0; j < (*level)[i].size(); j++) {
			//cout << string(to_string(i + 1).length() + 3, '\b') << string((to_string(i + 1).length() > 1 && to_string(i).length() == 1) ? 1 : 0, ' ') << i << "/" << level.size();
			//cout << "reading level " << i << endl;
			if (typeid(*(*level)[i][j]) == typeid(Dense)) {
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
				Dense* layer = dynamic_cast<Dense*>((*level)[i][j]);
				layer->weights = layerJ["weights"].get<denseW>();
				layer->beta = layerJ["beta"].get<denseA>();
				layer->gamma = layerJ["gamma"].get<denseA>();
				layer->avgM = layerJ["mean"].get<denseA>();
				layer->avgVar = layerJ["variance"].get<denseA>();
				layer->converged = true;
				inFile.close();
			}
			else if (typeid(*(*level)[i][j]) == typeid(Conv)) {
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
				Conv* layer = dynamic_cast<Conv*>((*level)[i][j]);
				layer->kernel = layerJ["kernel"].get<std::vector<convA>>();
				layer->beta = layerJ["beta"].get<std::vector<float>>();
				layer->gamma = layerJ["gamma"].get<std::vector<float>>();
				layer->biases = layerJ["bias"].get<std::vector<float>>();
				layer->avgM = layerJ["mean"].get<std::vector<float>>();
				layer->avgVar = layerJ["variance"].get<std::vector<float>>();
				layer->converged = true;
				inFile.close();
			}
			else if (typeid(*(*level)[i][j]) == typeid(Input)) {
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
				Input* layer = dynamic_cast<Input*>((*level)[i][j]);
				*batchMean = layerJ["batchMean"].get<float>();
				inFile.close();
			}
			else if (typeid(*(*level)[i][j]) == typeid(MultiDense)) {
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
				MultiDense* layer = dynamic_cast<MultiDense*>((*level)[i][j]);
				layer->avgM = layerJ["avgM"].get<float>();
				layer->avgVar = layerJ["avgVar"].get<float>();
				layer->betaM = layerJ["betaM"].get<denseA>();
				layer->alphaM = layerJ["alphaM"].get<denseA>();
				layer->beta = layerJ["beta"].get<denseA>();
				layer->gamma = layerJ["gamma"].get<denseA>();
				layer->position = layerJ["position"].get<std::vector<std::vector<float>>>();
				inFile.close();
			}
			else if (typeid(*(*level)[i][j]) == typeid(MultiDenseInput)) {
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
				MultiDenseInput* layer = dynamic_cast<MultiDenseInput*>((*level)[i][j]);
				layer->position = layerJ["position"].get<std::vector<std::vector<float>>>();
				inFile.close();
			}
		}
	}
};*/