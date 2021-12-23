#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <thread>


int main(int argc, const char **argv) {
  // generate random data serially
  int N = std::atoi(argv[1]);
  thrust::host_vector<int> h_vec(N);
  std::generate(h_vec.begin(), h_vec.end(), [n = 1] () mutable { return n++; });

  // transfer to device and compute sum
  std::cout << "transfer ... " << std::endl;
  thrust::device_vector<int> d_vec = h_vec;
  int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

  std::chrono::milliseconds timespan(int(5e3));
  std::this_thread::sleep_for(timespan);
  std::cout << x << std::endl;
  return 0;
}