#ifndef _activationsH_
#define _activationsH_

#include <vector>
#include <stxxl/vector>

typedef std::vector<float> denseA;
typedef std::vector<std::vector<std::vector<float>>> convA;
typedef std::vector<std::vector<float>> denseW;
typedef std::vector<std::vector<float>> featureMap;

typedef std::vector<float> vectorf;
typedef std::vector<std::vector<float>> vectorff;
typedef std::vector<std::vector<std::vector<float>>> convAL;
typedef std::vector<convA> vconvAL;

#endif