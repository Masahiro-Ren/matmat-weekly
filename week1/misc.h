#pragma once

#include <type_traits>
#include <chrono>
#include <random>
#include <cmath>

class Timer {
public:
void start();
double get() const;

private:
    using clock_t = std::chrono::steady_clock;
    clock_t::time_point start_{clock_t::now()};
};

template<typename T>
class RandomValue {
    static_assert(std::is_arithmetic_v<T>,
                  "RandomValue<T> requires an arithmetic type "
                  "(int, long, float, double, ...)");
    using distribution_t = std::conditional_t<
                            std::is_integral_v<T>,
                            std::uniform_int_distribution<T>,
                            std::uniform_real_distribution<T>>;
public:
    RandomValue() : engine_(std::random_device{}()) {}

    explicit RandomValue(std::mt19937_64::result_type seed) : engine_(seed) {}

    void setrange(T begin, T end)
    {
        dist_ = distribution_t(begin, end);
    }

    T getval()
    {
        return dist_(engine_);
    }
private:
    std::mt19937_64 engine_;
    distribution_t dist_{};
};

template<typename T>
void fill_random(T* a, std::size_t N, T lo, T hi)
{
    #pragma omp parallel
    {
        RandomValue<T> rng;
        rng.setrange(lo, hi);
        #pragma omp for
        for(std::size_t i = 0; i < N; i++)
            a[i] = rng.getval();
    }
}

template<typename T>
T checksum(const T* a, std::size_t N)
{
    T sum = T(0);

    #pragma omp parallel for reduction(+:sum)
    for(std::size_t i = 0; i < N; i++) sum += a[i];

    return sum;
}

template<typename T>
T max_abs_diff(const T* a, const T* b, std::size_t N)
{
    T m = T(0);

    #pragma omp parallel for reduction(max:m)
    for(std::size_t i = 0; i < N; i++)
        m = std::max(m, std::abs(a[i] - b[i]));
    
    return m;
}