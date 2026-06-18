#include "misc.h"

void Timer::start()
{
    start_ = clock_t::now();
}

double Timer::get() const
{
    const auto now = clock_t::now();
    return std::chrono::duration<double>(now - start_).count();
}
