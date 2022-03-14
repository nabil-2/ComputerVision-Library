#include <iostream>
#include <limits>

#include <stxxl/vector>
#include <stxxl/random>
#include <stxxl/sort>
#include <filesystem>
#include "deepNet.h"

float fct(float x);

int main() {
	std::string type = "CNN";
	if (type == "CNN") {
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
		layerIx = network.addLayer("Conv", 128, levelIx - 1, layerIx, 1, 1, 0); //27x27 //128 (Bottleneck)
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
		/*network.intialiseParameters();

		Trainer trainer(&network);
		trainer.setBatchSize(8);
		for (unsigned int i = 0; i < 10; i++) {
			trainer.train("C:\\Users\\admlocal\\Desktop\\KI\\Files\\CNN\\training", i);
		}
		trainer.finish();
		network.exportNet("converged.txt");*/
		//trainer.test("F:\\server\\results", "D:\\stxxl\\CNNv4\\build\\local\\Release\\structured");

		network.importNet("C:\\Users\\admlocal\\Desktop\\KI\\Files\\CNN\\converged.txt");
		vconvAL inputs;
		bool testImg = true;
		if (testImg) {
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/pen1.jpg"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/pen2.jpg"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/table1.jpg"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/table2.jpg"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/remote1.jpg"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/remote2.jpg"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/tel1.jpg"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_test/tel2.jpg"));
		} else {
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_train/pen.JPEG"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_train/table.JPEG"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_train/remote.JPEG"));
			inputs.push_back(FileManager::getInput("C:/Users/admlocal/Desktop/KI/Files/CNN/images_train/television.JPEG"));
		}
		network.setInput(&inputs);
		std::vector<std::vector<std::string>> result = network.predict();
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				std::cout << result[i][j] << std::endl;
			}
			std::cout << "-----------" << std::endl;
		}/**/
	}
	else if (type == "NN") {
		int inSize[3] = { 28, 28, 1 };
		Network network;
		int levelIx = network.addLevel();
		int layerIx = network.addLayer("Input", inSize);
		levelIx = network.addLevel();
		layerIx = network.addLayer("Flatten", levelIx - 1, layerIx);
		//levelIx = network.addLevel();
		//layerIx = network.addLayer("MultiDenseInput", levelIx - 1, layerIx);
		for (unsigned int i = 0; i < 3; i++) {
			levelIx = network.addLevel();
			layerIx = network.addLayer("Dense", 512, levelIx - 1, layerIx);
		}
		levelIx = network.addLevel();
		layerIx = network.addLayer("Dense", 10, levelIx - 1, layerIx); //Output: 10 Classes
		levelIx = network.addLevel();
		layerIx = network.addLayer("Softmax", levelIx - 1, layerIx);

		network.setActivationFunction("ReLU");
		network.setCostFunction("cross-entropy");
		/*network.intialiseParameters();

		Trainer trainer(&network);
		trainer.setBatchSize(64);
		for (unsigned int i = 0; i < 1; i++) {
			trainer.train("C:\\Users\\admlocal\\Desktop\\KI\\Files\\NN\\MNIST\\training", i);
		}
		trainer.finish();*/
		//trainer.test("results", "MNIST\\testing");

		network.importNet("C:\\Users\\admlocal\\Desktop\\KI\\Project\\CNN v3\\results\\converged459;0.txt");
		vconvAL in;
		in.reserve(10);
		for (unsigned int i = 0; i < 10; i++) {
			in.push_back(FileManager::getInput("C:\\Users\\admlocal\\Desktop\\KI\\Files\\NN\\test\\" + std::to_string(i) + ".png"));
		}
		network.setInput(&in);
		std::vector<std::vector<float>> result = network.predictVal();
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				std::cout << j << ": " << result[i][j] * 100 << "%" << std::endl;
			}
			std::cout << "----------" << std::endl;
		}
	}
	else {
		int inSize[3] = { 1, 1, 1 };
		Network network;
		int levelIx = network.addLevel();
		int layerIx = network.addLayer("Input", inSize);
		levelIx = network.addLevel();
		layerIx = network.addLayer("Flatten", levelIx - 1, layerIx);
		//levelIx = network.addLevel();
		//layerIx = network.addLayer("MultiDenseInput", levelIx - 1, layerIx);
		for (unsigned int i = 0; i < 1; i++) {
			levelIx = network.addLevel();
			layerIx = network.addLayer("Dense", 32, levelIx - 1, layerIx);
		}
		levelIx = network.addLevel();
		layerIx = network.addLayer("Dense", 1, levelIx - 1, layerIx);

		network.setActivationFunction("ReLU");
		network.setCostFunction("mean-squared");
		network.intialiseParameters();

		int batchS = 16;
		std::vector<convA> inputs;
		std::vector<denseA> outputs;
		int epochs = 50000;
		for (unsigned int j = 0; j < epochs; j++) {
			std::cout << "epoch " << j + 1 << ", ";
			srand(time(NULL));
			for (unsigned int i = 0; i < batchS; i++) {
				inputs.push_back(convA());
				inputs[i].push_back(vectorff());
				float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				inputs[i][0].push_back({ r * 4 - 2 });
				outputs.push_back(vectorf());
				outputs[i].push_back(fct(inputs[i][0][0][0]));
			}
			network.epoch(inputs, outputs);
			inputs = vconvAL();
			outputs = std::vector<denseA>();
		}
		network.exportNet("converged.txt");
	}

	//system("pause");
	return 0;
}

float fct(float x) {
	return 0.5 * pow(x, 1/3) + 1.7;
	return sin(x);
}