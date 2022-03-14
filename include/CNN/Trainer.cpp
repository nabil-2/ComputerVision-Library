#include "Trainer.h"
#include "FileManager.h"
#include <filesystem>
#include <iostream>
#include <regex>

//using namespace std;

Trainer::Trainer(Network* net) : network(net) {};
void Trainer::setBatchSize(int size) {
	if (size > 0) batchSize = size;
};
void Trainer::train(std::string path, int iteration) {
    system("mkdir results");
    network->calcBatchMean(128, path);
    int i = 1,
        epoch = 0;
    std::vector<std::vector<std::string>> data(0);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        //if (epoch == 5) break;
        if (!std::filesystem::is_directory(entry.path())) {
            std::string path{ entry.path().u8string() };
            std::string label, filename;
            std::regex regex1("\\\\");
            std::vector<std::string> out1(
                std::sregex_token_iterator(path.begin(), path.end(), regex1, -1),
                std::sregex_token_iterator()
            );
            filename = out1[out1.size() - 1];
            std::regex regex2("\\_");
            std::vector<std::string> out2(
                std::sregex_token_iterator(filename.begin(), filename.end(), regex2, -1),
                std::sregex_token_iterator()
            );
            label = out2[1];
            data.push_back(std::vector<std::string>({path, label}));
            if (i == batchSize) {
                std::cout << "epoch: " + std::to_string(epoch + 1) + ", " + "iteration " + std::to_string(iteration + 1) + ", ";
                //system("pause");
                /*data = { 
                    {"D:\\MNIST\\2\\structured\\00sp864jl8_0_458.png", "0"},
                    {"D:\\MNIST\\2\\structured\\00javjzbm4_1_14967.png", "1"},
                    {"D:\\MNIST\\2\\structured\\00y3b38fwy_2_17123.png", "2"},
                    {"D:\\MNIST\\2\\structured\\0ab6qfh97b_3_18977.png", "3"},
                    {"D:\\MNIST\\2\\structured\\0a5wmkhe8b_4_32673.png", "4"},
                    {"D:\\MNIST\\2\\structured\\0adgnhrtd6_5_5240.png", "5"},
                    {"D:\\MNIST\\2\\structured\\00qx7hh0pj_6_7744.png", "6"},
                    {"D:\\MNIST\\2\\structured\\00uwmo01ev_7_49615.png", "7"},
                    {"D:\\MNIST\\2\\structured\\0a6ayxevpa_8_26043.png", "8"},
                    {"D:\\MNIST\\2\\structured\\0aj2ymvtdd_9_482.png", "9"}
                };*/
                /*data = {
                    {"D:\\MNIST\\2\\structured\\0a6ayxevpa_8_26043.png", "8"}
                };*/
                network->epoch(&data);
                data = std::vector<std::vector<std::string>>(0);
                epoch++;
                i = 0;
                if (epoch % 1 == 0) network->exportNet("results/converged" + std::to_string(epoch) + ";" + std::to_string(iteration) + ".txt");
            }
            i++;
        }
    }
};
void Trainer::test(std::string networkPath, std::string testPath) {
    int sequenceSize = 4,
        maximumImgs = 4;
    vconvAL data;
    std::vector<std::string> labels;
    std::vector<int> quantity;
    std::vector<std::vector<std::vector<float>>> testRes; //testRes[iteration][epoch][acc/conf]
    for (const auto& entry : std::filesystem::recursive_directory_iterator(networkPath)) {
        if (!std::filesystem::is_directory(entry.path())) {
            std::string netPath{ entry.path().u8string() };
            std::string filename;
            std::regex regex1("\\\\");
            std::vector<std::string> out1(
                std::sregex_token_iterator(netPath.begin(), netPath.end(), regex1, -1),
                std::sregex_token_iterator()
            );
            filename = out1[out1.size() - 1];
            std::regex_iterator<std::string::iterator>::regex_type reg("[0-9]+");
            std::regex_iterator<std::string::iterator> next(filename.begin(), filename.end(), reg), end;
            std::vector<int> res;
            for (; next != end; next++) {
                res.push_back(std::stoi(next->str()));
            }
            int epoch = res[0],
                iteration = res[1];
            if (iteration >= quantity.size()) {
                for (unsigned int i = quantity.size(); i <= iteration; i++) {
                    quantity.push_back(0);
                }
            }
            quantity[iteration] = quantity[iteration] < epoch ? epoch : quantity[iteration];
        }
    }
    for (unsigned int i = 0; i < quantity.size(); i++) {
        testRes.push_back(std::vector<std::vector<float>>());
        for (unsigned int j = 0; j < quantity[i]; j++) {
            testRes[i].push_back(std::vector<float>());
        }
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(networkPath)) {
        if (!std::filesystem::is_directory(entry.path())) {
            std::string netPath{ entry.path().u8string() };
            //std::cout << netPath << std::endl;
            network->importNet(netPath);
            int i = 0;
            std::vector<float> sucess;
            int imgN = 0;
            for (const auto& entry2 : std::filesystem::recursive_directory_iterator(testPath)) {
                if (!std::filesystem::is_directory(entry2.path())) {
                    imgN++;
                    i++;
                    std::string imgPath{ entry2.path().u8string() };
                    std::string label, filename;
                    std::regex regex1("\\\\");
                    std::vector<std::string> out1(
                        std::sregex_token_iterator(imgPath.begin(), imgPath.end(), regex1, -1),
                        std::sregex_token_iterator()
                    );
                    filename = out1[out1.size() - 1];
                    std::regex regex2("\\_");
                    std::vector<std::string> out2(
                        std::sregex_token_iterator(filename.begin(), filename.end(), regex2, -1),
                        std::sregex_token_iterator()
                    );
                    label = out2[1];
                    data.push_back(FileManager::getInput(imgPath));
                    labels.push_back(label);
                    if (i == sequenceSize) {
                        network->setInput(&data);
                        std::vector<std::vector<float>> results = network->predictVal();
                        for (unsigned int j = 0; j < results.size(); j++) {
                            std::string outIx = std::to_string(std::max_element(results[j].begin(), results[j].end()) - results[j].begin());
                            /*if (outIx == labels[j]) { //imgs: outIx of synsetID == labels[j]
                                float out = *std::max_element(results[j].begin(), results[j].end());
                                sucess.push_back(100*out);
                            }*/
                            if (network->synsetID[stoi(outIx)][0] == labels[j]) {
                                float out = *std::max_element(results[j].begin(), results[j].end());
                                sucess.push_back(100 * out);
                            }
                        }
                        i = 0;
                        data.clear();
                        labels.clear();
                        data.shrink_to_fit();
                        labels.shrink_to_fit();
                        if (imgN >= maximumImgs) break;
                    }
                }
            }
            std::string filename;
            std::regex regex1("\\\\");
            std::vector<std::string> out1(
                std::sregex_token_iterator(netPath.begin(), netPath.end(), regex1, -1),
                std::sregex_token_iterator()
            );
            filename = out1[out1.size() - 1];
            std::regex_iterator<std::string::iterator>::regex_type reg("[0-9]+");
            std::regex_iterator<std::string::iterator> next(filename.begin(), filename.end(), reg), end;
            std::vector<int> res;
            for (; next != end; next++) {
                res.push_back(std::stoi(next->str()));
            }
            int epoch = res[0],
                iteration = res[1];
            float accuracy = 100 * ((float)sucess.size()) / ((float)imgN);
            float confidence = 0;
            for (unsigned int j = 0; j < sucess.size(); j++) {
                confidence += sucess[j];
            }
            if (sucess.size() > 0) confidence /= sucess.size();
            std::cout << "ep: " << epoch << ", it: " << iteration << ", acc: " << accuracy << ", conf: " << confidence << std::endl;
            testRes[iteration][epoch-1] = { accuracy, confidence };
        }
    }
    std::string acc = "",
                conf = "";
    for (unsigned int i = 0; i < testRes.size(); i++) {
        for (unsigned int j = 0; j < testRes[i].size(); j++) {
            acc += std::to_string(testRes[i][j][0]) + ";";
            conf += std::to_string(testRes[i][j][1]) + ";";
        }
    }
    std::ofstream out;
    out.open("testResults.txt");
    out << acc + "\n" + conf;
    out.close();
};
void Trainer::finish() {
    network->finishTraining();
};