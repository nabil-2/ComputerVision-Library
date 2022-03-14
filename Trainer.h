#ifndef _TrainerH_
#define _TrainerH_

#include "Network.h"

class Trainer {
private:
	int batchSize = 64;
	Network* network;
public:
	Trainer(Network *network);
	void setBatchSize(int size);
	void train(std::string dataFolder, int iteration);
	void test(std::string dataFolder, std::string networkFolder);
	void finish();
};

#endif