#include "Input.h"

//using namespace std;

Input::Input(int* size_) : size(size_) {
	//activations = std::vector<convA>(1, convA(size[0], std::vector<std::vector<float>>(size[1], std::vector<float>(size[2]))));
	activations.push_back(convAL());
	for (unsigned int i = 0; i < size[0]; i++) {
		activations[0].push_back(vectorff());
		for (unsigned int j = 0; j < size[1]; j++) {
			activations[0][i].push_back(vectorf());
			for (unsigned int k = 0; k < size[2]; k++) {
				activations[0][i][j].push_back(0);
			}
		}
	}
	//activations.push_back(vector<);
	//convA(size[0], vector<vector<float>>(size[1], vector<float>(size[2])));
};
void Input::forward() {
};
void Input::backward() {};
void Input::initialiseParameters(std::string actF) {};