#ifndef _FileManagerH_
#define _FileManagerH_

#include "activations.h"
#include <string>


class FileManager {
public:
	static convA getInput(std::string imgPath);
	static convA grayscale(convA* image);
	static convA invert(convA* image);
	//static void exportNet(std::string filename, float batchMean, std::vector< std::vector<Layer*>>* level);
	//static void importNet(std::string filename, float* batchMean, std::vector< std::vector<Layer*>>* level);
};

#endif