#include "FileManager.h"
#include <iostream>

typedef std::vector<float> denseA;
typedef std::vector<std::vector<std::vector<float>>> convA;
typedef std::vector<std::vector<float>> denseW;
typedef std::vector<std::vector<float>> featureMap;

extern "C" {
	#define STB_IMAGE_IMPLEMENTATION
	#include "stb_image.h"
}

convA FileManager::getInput(std::string filename) {
	std::vector<std::vector<std::vector<float>>> val;
	int height, width, channels;
	unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
	for (int ch = 0; ch < channels; ch++) {
		int dataIx = ch;
		std::vector<float> chVals;
		for (int i = 0; i < height * width; i++) {
			chVals.push_back((float)static_cast<int>(data[dataIx]) / (float)255);
			dataIx += channels;
		}
		std::vector<std::vector<float>> tmp;
		for (int y = 0; y < height; y++) {
			tmp.push_back(std::vector<float>());
			for (int x = 0; x < width; x++) {
				int ix = y * width + x;
				tmp[y].push_back(chVals[ix]);
			}
		}
		val.push_back(tmp);
	}
	std::vector<std::vector<std::vector<float>>> val2(width, std::vector<std::vector<float>>(height, std::vector<float>(channels)));
	stbi_image_free(data);
	for (unsigned int i = 0; i < val.size(); i++) {
		for (unsigned int j = 0; j < val[i].size(); j++) {
			for (unsigned int k = 0; k < val[i][j].size(); k++) {
				val2[k][j][i] = val[i][j][k];
			}
		}
	}
	return val2;
}
convA FileManager::grayscale(convA* image) {
	if ((*image)[0][0].size() == 1) {
		return *image;
	}
	std::vector<std::vector<std::vector<float>>> result(image->size(), std::vector<std::vector<float>>((*image)[0].size(), std::vector<float>(1)));
	for (unsigned int i = 0; i < image->size(); i++) {
		for (unsigned int j = 0; j < (*image)[i].size(); j++) {
			result[i][j][0] = (*image)[i][j][0];
		}
	}
	return result;
};
convA FileManager::invert(convA* image) {
	std::vector<std::vector<std::vector<float>>> result(image->size(), std::vector<std::vector<float>>((*image)[0].size(), std::vector<float>((*image)[0][0].size())));
	for (unsigned int i = 0; i < image->size(); i++) {
		for (unsigned int j = 0; j < (*image)[i].size(); j++) {
			for (unsigned int k = 0; k < (*image)[i][j].size(); k++) {
				result[i][j][k] = 1 - (*image)[i][j][k];
			}
		}
	}
	return result;
};

/*void FileManager::exportNet(std::string filename, float batchMean, std::vector<std::vector<Layer*>>* level) {
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