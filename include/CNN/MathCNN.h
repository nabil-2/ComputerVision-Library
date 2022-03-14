#ifndef _MathCNNH_
#define _MathCNNH_

#include <string>
#include <vector>
#include <random>
#include "activations.h"
//using namespace std;

class MathCNN {
private:
	int static factorial(int n);
	float static ck(float k);
	float static erf_1(float x);
	const std::vector<std::string> alphabet = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
	int position(std::string letter);
	float static scalarProduct(const std::vector<float>* vec1, const std::vector<float>* vec2);
public:
	MathCNN();
	float static getMeanDistance(float sigmaW);
	float static getDistance(float sigmaW, float uniformNo);
	std::vector<float> static MVProduct(const std::vector<std::vector<float>>* matrix, const std::vector<float>* vector);
	void static MVaddColumn(std::vector<std::vector<float>>* matrix, const std::vector<float>* vector);
	void static MVmultiplyColumn(std::vector<std::vector<float>>* matrix, const std::vector<float>* vector);
	std::vector<std::vector<float>> static MMProduct(const std::vector<std::vector<float>>* matrix1, const std::vector<std::vector<float>>* matrix2);
	float static getRandomNormal(const float* mean, const float* dev, std::default_random_engine* generator);
	std::vector<float> static addVec(std::vector<float>* vec1, std::vector<float>* vec2);
	std::vector<float> static subtractVec(std::vector<float>* vec1, std::vector<float>* vec2);
	float static distance(std::vector<float>* p1, std::vector<float>* p2);
	float ReLU(float z);
	static void ReLU(float* z);
	static void ReLU(std::vector<float>* z);
	static void ReLU(std::vector<std::vector<float>>* z);
	static void ReLU_(std::vector<float>* z);
	static std::vector<convA> ReLU(std::vector<convA>* z);
	static std::vector<convA> ReLU_(std::vector<convA>* z);
	static void ReLU(vconvAL* z, vconvAL* writeInto);
	static void ReLU_(vconvAL* z, vconvAL* writeInto);
	static void softmax(const std::vector<float>* in, std::vector<float>* out);
	static void dotProduct(const std::vector<float>* in1, const std::vector<float>* in2, std::vector<float>* result);
	static std::vector<float> dotProduct(const std::vector<float>* in1, const std::vector<float>* in2);
	std::vector<std::vector<float>> static transposeM(const std::vector<std::vector<float>>* matrix);
	std::string encode(std::string input);
	std::string decode(std::string input);
	featureMap static conv(convAL *layer, convA *kernel, int stride);
	featureMap static conv(convAL* layer, convA* kernel, int stride, int padding);
	//featureMap static conv(convA *layer, convA *kernel, int stride);
	//featureMap static conv(convA* layer, convA* kernel, int stride, int padding);
	int static convSize(int x, int s, int p, int k);
	static void multiplyElementwise(std::vector<convA> *u, convA *gamma);
	static void addElementwise(std::vector<convA>* u, convA* beta);
	static void retransformBN(vconvAL* u, std::vector<float>* gamma, std::vector<float>* beta);
	static void addActMapwise(vconvAL* u, std::vector<float>* beta);
	static convAL dotProduct(convAL* mtrx1, convAL* mtrx2);
	static convA rotate180(convA* kernel);
	static convAL asStxxl(convA* in);
	static convA aStd(convAL* in);
	static vconvAL asStxxl(std::vector<convA>* in);
};

#endif