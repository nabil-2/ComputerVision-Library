#include <iostream>
#include <limits>

#include <stxxl/vector>
#include <stxxl/random>
#include <stxxl/sort>
#include <filesystem>
#include "CNN/deepNet.h"

int main() {
    //CNN
	int inSize[3] = { 196, 196, 3 };
	Network network;
	int levelIx = network.addLevel();
	int layerIx = network.addLayer("Input", inSize);

	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 64, levelIx - 1, layerIx, 3, 1, 0); //194x194 //64
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 64, levelIx - 1, layerIx, 3, 1, 0); //192x192 //64
	levelIx = network.addLevel();
	layerIx = network.addLayer(2, "Pooling", levelIx - 1, layerIx); //97x97
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 128, levelIx - 1, layerIx, 3, 1, 0); //95x95 //128
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 128, levelIx - 1, layerIx, 3, 1, 0); //93x93 //128
	levelIx = network.addLevel();
	layerIx = network.addLayer(3, "Pooling", levelIx - 1, layerIx); //31x31
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 256, levelIx - 1, layerIx, 3, 1, 0); //29x29 //256
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 256, levelIx - 1, layerIx, 3, 1, 0); //27x27 //256
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 128, levelIx - 1, layerIx, 1, 1, 0); //27x27 //64 (Bottleneck)
	levelIx = network.addLevel();
	layerIx = network.addLayer(3, "Pooling", levelIx - 1, layerIx); //9x9
	levelIx = network.addLevel();
	layerIx = network.addLayer("Flatten", levelIx - 1, layerIx); //1x5184
	levelIx = network.addLevel();
	layerIx = network.addLayer("Dense", 2048, levelIx - 1, layerIx);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Dense", 2048, levelIx - 1, layerIx);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Dense", 2048, levelIx - 1, layerIx);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Dense", 4, levelIx - 1, layerIx); //Output: 4 Classes
	levelIx = network.addLevel();
	layerIx = network.addLayer("Softmax", levelIx - 1, layerIx);

	network.setActivationFunction("ReLU");
	network.setCostFunction("cross-entropy");
	network.intialiseParameters();

	Trainer trainer(&network);
	trainer.setBatchSize(32);
	for (unsigned int i = 0; i < 10; i++) {
		trainer.train("structured", i);
	}
	trainer.finish();    
	network.exportNet("converged.txt");
	trainer.test("results", "testData");
    
    //Neural Network
    int inSize[3] = { 28, 28, 1 };
	Network network;
	int levelIx = network.addLevel();
	int layerIx = network.addLayer("Input", inSize);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Flatten", levelIx - 1, layerIx);
	levelIx = network.addLevel();
	layerIx = network.addLayer("MultiDenseInput", levelIx - 1, layerIx);
	for (unsigned int i = 0; i < 2; i++) {
		levelIx = network.addLevel();
		layerIx = network.addLayer("MultiDense", 128, levelIx - 1, layerIx);
	}
	levelIx = network.addLevel();
	layerIx = network.addLayer("MultiDense", 10, levelIx - 1, layerIx); //Output: 10 Classes
	levelIx = network.addLevel();
	layerIx = network.addLayer("Softmax", levelIx - 1, layerIx);

	network.setActivationFunction("ReLU");
	network.setCostFunction("cross-entropy");
	network.intialiseParameters();

    Trainer trainer(&network);
	trainer.setBatchSize(64);
	for (unsigned int i = 0; i < 1; i++) {
		trainer.train("MNIST\\structured", i);
	}
	trainer.finish();
	network.exportNet("converged.txt");
	trainer.test("results", "MNIST\\testing");

    system("pause");
    return 0;
}