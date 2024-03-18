
#include <vector>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum class ExecutionContext { CPU, GPU };
using namespace std;

template<typename T>
__global__ void convolutionKernel(T* data, T* filter, T* output, int dataSize, int filterSize) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < dataSize + filterSize - 1) {
    int outputIndex = tid;
    T sum = 0.0;
    for (int i = 0; i < filterSize; ++i) {
      int dataIndex = tid - i;
      if (dataIndex >= 0 && dataIndex < dataSize) {
        sum += data[dataIndex] * filter[i];
      }
    }
    output[outputIndex] = sum;
  }
}

template <class T>
class Conv
{
public:
  explicit Conv(vector<T>& d, vector<T>& f) : data(d), filter(f) {}

  template<ExecutionContext cpuOrGPU>
  vector<T> DoConv()
  {
    if constexpr (cpuOrGPU == ExecutionContext::CPU)
    {
      return ConvCPU();
    }
    else if constexpr (cpuOrGPU == ExecutionContext::GPU) {
      return ConvGPU();
    }
  }

private:
  vector<T> ConvCPU()
  {
    vector<T> output(data.size() + filter.size() - 1);
    for (int i = 0; i < output.size(); ++i)
    {
      for (int filterIndex = filter.size() - 1; filterIndex >= 0; filterIndex--) {
        auto dataIndex = i - filterIndex;
        output[i] += (dataIndex >= 0 && dataIndex < data.size()) ? filter[filterIndex] * data[dataIndex] : 0;
      }
    }
    return output;
  }

  vector<T> ConvGPU()
  {
    int cuDataSize = data.size();
    int cuFilterSize = filter.size();
    vector<T> result(cuDataSize + cuFilterSize - 1);

    T* cuData, * cuFilter, * output;

    cudaMalloc((void**)&cuData, cuDataSize * sizeof(T));
    cudaMalloc((void**)&cuFilter, cuFilterSize * sizeof(T));
    cudaMalloc((void**)&output, (cuDataSize + cuFilterSize - 1) * sizeof(T));

    cudaMemcpy(cuData, data.data(), cuDataSize * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(cuFilter, filter.data(), cuFilterSize * sizeof(T), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (cuDataSize + cuFilterSize - 1 + blockSize - 1) / blockSize;

    convolutionKernel << <numBlocks, blockSize >> > (cuData, cuFilter, output, cuDataSize, cuFilterSize);

    cudaMemcpy(result.data(), output, (cuDataSize + cuFilterSize - 1) * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(cuData);
    cudaFree(cuFilter);
    cudaFree(output);

    return result;
  }

  vector<T>& data;
  vector<T>& filter;
};

template<class T>
void print(vector<T> numbers)
{
  for (auto n : numbers) { cout << n << " "; } cout << endl;
}

int main()
{
  // test CPU (int)
  vector<int> input1(10, 1), filter1(3, 1);
  Conv<int> conv1(input1, filter1);
  auto out1 = conv1.DoConv<ExecutionContext::CPU>();
  print(out1); // 1 2 3 3 3 3 3 3 3 3 2 1

  // test GPU (int)
  vector<int> input2(10, 1); vector<int> filter2(3, 2);
  Conv<int> conv2(input2, filter2);
  auto out2 = conv2.DoConv<ExecutionContext::GPU>();
  print(out2); // 2 4 6 6 6 6 6 6 6 6 4 2

  // test CPU (double)
  vector<double> input3(10, 1.1), filter3(3, 3.0);
  Conv<double> conv3(input3, filter3);
  auto out3 = conv3.DoConv<ExecutionContext::CPU>();
  print(out3); // 3.3 6.6 9.9 9.9 9.9 9.9 9.9 9.9 9.9 9.9 6.6 3.3

  // test GPU (double)
  vector<double> input4(10, 1.1), filter4(3, 4.0);
  Conv<double> conv4(input4, filter4);
  auto out4 = conv4.DoConv<ExecutionContext::GPU>();
  print(out4); // 4.4 8.8 13.2 13.2 13.2 13.2 13.2 13.2 13.2 13.2 8.8 4.4

  return 0;
}
