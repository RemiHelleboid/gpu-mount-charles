#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <thread>

struct point {
    float                              m_x;
    float                              m_y;
    float                              m_z;
    thrust::default_random_engine      m_rng;
    thrust::normal_distribution<float> dist{0, 1};

    point() : m_x{0.0}, m_y{0.0}, m_z{0.0}, m_rng(0.0) {}
    point(uint seed) : m_x{0.0}, m_y{0.0}, m_z{0.0}, m_rng{seed} {}
    point(float x, float y, float z) : m_x{x}, m_y{y}, m_z{z}, m_rng{0} {}
    point(float x, float y, float z, uint seed) : m_x{x}, m_y{y}, m_z{z}, m_rng{seed} {}

    __host__ __device__ void brownian_motion(float dt) {
        m_x += sqrt(dt) * dist(m_rng);
        m_y += sqrt(dt) * dist(m_rng);
        m_z += sqrt(dt) * dist(m_rng);
    }
};

void export_coords(const std::string &filename, const std::vector<point> &vec) {
    std::ofstream my_file(filename);
    my_file << "X,Y,Z" << std::endl;
    for (auto &&my_point : vec) {
        my_file << my_point.m_x << "," << my_point.m_y << "," << my_point.m_z << "\n";
    }
    my_file.close();
}

void export_coords(const std::string &filename, const thrust::host_vector<point> &vec) {
    std::ofstream my_file(filename);
    my_file << "X,Y,Z" << std::endl;
    for (auto &&my_point : vec) {
        my_file << my_point.m_x << "," << my_point.m_y << "," << my_point.m_z << "\n";
    }
    my_file.close();
}

struct move_functor {
    double m_dt;
    move_functor(double dt) : m_dt{dt} {}
    __host__ __device__ point operator()(point &my_point) {
        my_point.brownian_motion(m_dt);
        return my_point;
    }
};

int main(int argc, const char **argv) {
    // generate random data serially
    std::size_t                N         = std::atoi(argv[1]);
    double                     step_time = std::atof(argv[2]);
    thrust::host_vector<point> h_vec(N);
    thrust::host_vector<int>   htmp_vec(N);

    std::generate(h_vec.begin(), h_vec.end(), [n = 1]() mutable {
        uint seed = std::random_device()();
        return point(seed);
    });
    std::generate(htmp_vec.begin(), htmp_vec.end(), [n = 1]() mutable { return n++; });

    // transfer to device and compute sum
    std::cout << "transfer ... " << std::endl;
    thrust::device_vector<point> d_vec = h_vec;
    thrust::device_vector<point> res_vec(d_vec);
    thrust::host_vector<point>   h_res_vec(h_vec.size());
    std::cout << "compute ... " << std::endl;
    move_functor motion{step_time};
    auto         start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < 1000; i++) {
        thrust::transform(d_vec.begin(), d_vec.end(), res_vec.begin(), motion);
        // h_vec = d_vec;
        // export_coords("traj/point_coords." + std::to_string(i) + ".csv", h_vec);
    }

    auto stop     = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << std::scientific << "Time of simulation GPU : " << duration.count() << " ns" << std::endl;

    auto start_serial = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < 1000; i++) {
        std::transform(h_vec.begin(), h_vec.end(), h_res_vec.begin(), motion);
    }
    auto stop_serial     = std::chrono::high_resolution_clock::now();
    auto duration_serial = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_serial - start_serial);
    std::cout << std::scientific << "Time of simulation CPU : " << duration_serial.count() << " ns" << std::endl;
    std::cout << std::scientific << "Time CPU / Time GPU = " << 1.0 * duration_serial / duration << std::endl;

    h_vec = d_vec;

    return 0;
}