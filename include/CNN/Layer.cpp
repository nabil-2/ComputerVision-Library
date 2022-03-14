#include "Layer.h"
#include <iostream>

void Layer::BatchNorm(std::vector<std::vector<float>>* z, denseA* mean, denseA* stddev, float delta) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			(*mean)[j] += (*z)[i][j];
		}
	}
	for (unsigned int j = 0; j < (*z)[0].size(); j++) {
		(*mean)[j] /= z->size();
	}
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			(*stddev)[j] += (float)pow((*z)[i][j] - (*mean)[j], 2);
		}
	}
	for (unsigned int j = 0; j < (*z)[0].size(); j++) {
		(*stddev)[j] /= z->size();
	}
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			(*z)[i][j] -= (*mean)[j];
			(*z)[i][j] /= sqrt((*stddev)[j] + delta);
		}
	}
};

void Layer::BatchNorm(std::vector<std::vector<float>>* z, denseA mean, denseA var, float delta) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			(*z)[i][j] -= mean[j];
			(*z)[i][j] /= sqrt(var[j] + delta);
		}
	}
};

void Layer::BatchNorm(vconvAL* z, std::vector<float>* mean, std::vector<float>* stddev2, float delta) {
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					(*mean)[l] += (*z)[i][j][k][l];
				}
			}
		}
	}
	int bxwxh = z->size() * (*z)[0].size() * (*z)[0][0].size();
	for (unsigned int l = 0; l < (*z)[0][0][0].size(); l++) {
		(*mean)[l] /= (float)bxwxh;
	}
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					(*stddev2)[l] += (float) pow((*z)[i][j][k][l] - (*mean)[l], 2);
				}
			}
		}
	}
	for (unsigned int l = 0; l < (*z)[0][0][0].size(); l++) {
		(*stddev2)[l] /= (float)bxwxh;
	}
	//cout << "Mean: " << (*mean)[0] << endl;
	//cout << "Variance: " << (*stddev2)[0] << endl;
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					(*z)[i][j][k][l] -= (*mean)[l];
					(*z)[i][j][k][l] /= sqrt((*stddev2)[l] + delta);
				}
			}
		}
	}
};

void Layer::BatchNorm(vconvAL* z, std::vector<float> mean, std::vector<float> var, float delta) {
	/*vector<float> m((*z)[0][0][0].size());
	vector<float> s((*z)[0][0][0].size());
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					m[l] += (*z)[i][j][k][l];
				}
			}
		}
	}
	int bxwxh = z->size() * (*z)[0].size() * (*z)[0][0].size();
	for (unsigned int l = 0; l < (*z)[0][0][0].size(); l++) {
		m[l] /= (float)bxwxh;
	}
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					s[l] += (float)pow((*z)[i][j][k][l] - m[l], 2);
				}
			}
		}
	}
	for (unsigned int l = 0; l < (*z)[0][0][0].size(); l++) {
		s[l] /= (float)bxwxh;
	}

	cout << "mean: " << m[0] << endl;
	cout << "variance: " << s[0] << endl;*/
	for (unsigned int i = 0; i < z->size(); i++) {
		for (unsigned int j = 0; j < (*z)[i].size(); j++) {
			for (unsigned int k = 0; k < (*z)[i][j].size(); k++) {
				for (unsigned int l = 0; l < (*z)[i][j][k].size(); l++) {
					//(*z)[i][j][k][l] -= m[i];
					//(*z)[i][j][k][l] /= sqrt(s[i] + delta);
					(*z)[i][j][k][l] -= mean[l];
					(*z)[i][j][k][l] /= sqrt(var[l] + delta);
				}
			}
		}
	}
};