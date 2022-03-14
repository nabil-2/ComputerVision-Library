/***************************************************************************
 *  local/test1.cpp
 *
 *  This is an example file included in the local/ directory of STXXL. All .cpp
 *  files in local/ are automatically compiled and linked with STXXL by CMake.
 *  You can use this method for simple prototype applications.
 *
 *  Part of the STXXL. See http://stxxl.sourceforge.net
 *
 *  Copyright (C) 2013 Timo Bingmann <tb@panthema.net>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 **************************************************************************/

#include <iostream>
#include <limits>

#include <stxxl/vector>
#include <stxxl/random>
#include <stxxl/sort>
#include <filesystem>
#include "CNN/deepNet.h"


/*struct my_less_int : std::less<int>
{
    int min_value() const { return std::numeric_limits<int>::min(); }
    int max_value() const { return std::numeric_limits<int>::max(); }
};*/

int main()
{
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
	layerIx = network.addLayer("Dense", 4, levelIx - 1, layerIx); //Output: 1000 Classes
	levelIx = network.addLevel();
	layerIx = network.addLayer("Softmax", levelIx - 1, layerIx);

	network.setActivationFunction("ReLU");
	network.setCostFunction("cross-entropy");
	//network.intialiseParameters();

	Trainer trainer(&network);
	/*trainer.setBatchSize(32);
	//trainer.train("structured", "");
	for (unsigned int i = 0; i < 10; i++) {
		trainer.train("structured", i);
		//trainer.train("F:\\test\\structured", "", i);
	}
	//trainer.train("F:\\ImageNet\\Localozation 2017\\imagenet-object-localization-challenge\\imagenet_object_localization_patched2019\\ILSVRC\\Data\\CLS-LOC\\structured", "");
	trainer.finish();
	std::cout << "exporting..." << std::endl;
	network.exportNet("converged.txt");*/
	trainer.test("F:\\server\\results", "D:\\stxxl\\CNNv4\\build\\local\\Release\\structured");
	system("pause");

	/*network.importNet("C:\\Users\\nabil\\OneDrive\\Desktop\\converged108;2.txt");
	std::vector<convA> in = {
		FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\pen.jpg"),
		FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\pen2.JPEG"),
		FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\table.jpg"),
		FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\tel.jpg"),
		FileManager::getInput("F:\\test\\test.JPEG"),
		FileManager::getInput("F:\\test\\test2.JPEG"),
		FileManager::getInput("F:\\test\\test3.JPEG"),
		FileManager::getInput("F:\\test\\test4.JPEG"),
		FileManager::getInput("F:\\test\\test.JPEG"),
		FileManager::getInput("F:\\test\\test2.JPEG"),
		FileManager::getInput("F:\\test\\test3.JPEG"),
		FileManager::getInput("F:\\test\\test4.JPEG")
	};
	network.setInput(&in);
	std::vector<std::vector<std::string>> result = network.predict();
	for (unsigned int i = 0; i < result.size(); i++) {
		for (unsigned int j = 0; j < result[i].size(); j++) {
			std::cout << result[i][j] << std::endl;
		}
		std::cout << "-----------" << std::endl;
	}
	system("pause");*/

	/*int inSize[3] = { 28, 28, 1 };
	Network network;
	int levelIx = network.addLevel();
	int layerIx = network.addLayer("Input", inSize);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 8, levelIx - 1, layerIx, 4, 1, 0); //25x25
	levelIx = network.addLevel();
	layerIx = network.addLayer("Conv", 16, levelIx - 1, layerIx, 5, 1, 0); //21x21
	levelIx = network.addLevel();
	layerIx = network.addLayer(3, "Pooling", levelIx - 1, layerIx); //7x7
	levelIx = network.addLevel();
	layerIx = network.addLayer("Flatten", levelIx - 1, layerIx);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Dense", 256, levelIx - 1, layerIx);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Dense", 256, levelIx - 1, layerIx);
	levelIx = network.addLevel();
	layerIx = network.addLayer("Dense", 10, levelIx - 1, layerIx); //Output: 10 Classes
	levelIx = network.addLevel();
	layerIx = network.addLayer("Softmax", levelIx - 1, layerIx);

	network.setActivationFunction("ReLU");
	network.setCostFunction("cross-entropy");
	network.intialiseParameters();

	Trainer trainer(&network);
	trainer.setBatchSize(64);
	//trainer.train("F:\\ImageNet\\Localozation 2017\\imagenet-object-localization-challenge\\imagenet_object_localization_patched2019\\ILSVRC\\Data\\CLS-LOC\\structured", "");
	for (unsigned int i = 0; i < 50; i++) {
		trainer.train("D:\\MNIST\\2\\structured", "", i);
	}
	network.exportNet("converged.txt");
	trainer.finish();
	system("pause");*/

	/*
	//network.importNet("results/converged10.txt");
	network.importNet("converged.txt");
	//convA img1 = FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test3.png");
	vconvAL in;
	in.push_back(FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test.png"));
	in.push_back(FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test.png"));
	in.push_back(FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test2.png"));
	network.setInput(&in);
	std::vector<std::vector<float>> result = network.predictVal();
	for (unsigned int i = 0; i < result.size(); i++) {
		for (unsigned int j = 0; j < result[i].size(); j++) {
			std::cout << j << ": " << result[i][j] * 100 << "%" << std::endl;
		}
		std::cout << "----------" << std::endl;
	}
	system("pause");*/

	/*int inSize[3] = { 28, 28, 1 };
		Network network;
		int levelIx = network.addLevel();
		int layerIx = network.addLayer("Input", inSize);
		levelIx = network.addLevel();
		layerIx = network.addLayer("Flatten", levelIx - 1, layerIx);
		//levelIx = network.addLevel();
		//layerIx = network.addLayer("MultiDenseInput", levelIx - 1, layerIx);
		for (unsigned int i = 0; i < 2; i++) {
			levelIx = network.addLevel();
			layerIx = network.addLayer("Dense", 128, levelIx - 1, layerIx);
		}
		levelIx = network.addLevel();
		layerIx = network.addLayer("Dense", 10, levelIx - 1, layerIx); //Output: 10 Classes
		levelIx = network.addLevel();
		layerIx = network.addLayer("Softmax", levelIx - 1, layerIx);

		network.setActivationFunction("ReLU");
		network.setCostFunction("cross-entropy");
		//network.intialiseParameters();

		Trainer trainer(&network);
		trainer.setBatchSize(64);
		//trainer.train("F:\\ImageNet\\Localozation 2017\\imagenet-object-localization-challenge\\imagenet_object_localization_patched2019\\ILSVRC\\Data\\CLS-LOC\\structured", "");
		for (unsigned int i = 0; i < 1; i++) {
			trainer.train("D:\\MNIST\\2\\structured", i);
		}
		//network.exportNet("converged.txt");
		trainer.finish();
		trainer.test("D:\\stxxl\\CNNv4\\build\\local\\Release\\results", "D:\\MNIST\\2\\testing");
		//trainer.test("results", "D:\\MNIST\\2\\testing");
		//trainer.test("F:\\server\\results", "D:\\MNIST\\2\\testing");
		system("pause");*/

		/*
		//network.importNet("results/converged10.txt");
		network.importNet("converged.txt");
		//convA img1 = FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test3.png");
		vconvAL in;
		in.push_back(FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test.png"));
		in.push_back(FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test.png"));
		in.push_back(FileManager::getInput("C:\\Users\\nabil\\OneDrive\\Desktop\\test2.png"));
		network.setInput(&in);
		std::vector<std::vector<float>> result = network.predictVal();
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				std::cout << j << ": " << result[i][j] * 100 << "%" << std::endl;
			}
			std::cout << "----------" << std::endl;
		}
		system("pause");*/

    /*
	// create vector
    stxxl::VECTOR_GENERATOR<int>::result vector;

    // fill vector with random integers
    {
        stxxl::scoped_print_timer
            timer("write random numbers", 100 * 1024 * 1024 * sizeof(int));

        stxxl::random_number32 random;

        for (size_t i = 0; i < 100 * 1024 * 1024; ++i) {
            vector.push_back(random());
        }
    }

    // sort vector using 16 MiB RAM
    {
        stxxl::scoped_print_timer
            timer("sorting random numbers", 100 * 1024 * 1024 * sizeof(int));

        stxxl::sort(vector.begin(), vector.end(), my_less_int(), 16 * 1024 * 1024);
    }

    // output first and last items:
    std::cout << vector.size() << " items sorted ranging from "
              << vector.front() << " to " << vector.back() << std::endl;
*/
    return 0;
}
