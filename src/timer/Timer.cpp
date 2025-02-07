#include "Timer.hpp"

Timer::Timer() : start_(std::chrono::high_resolution_clock::now()) {
}

int64_t Timer::GetMilliseconds() const {
    auto elapsed = std::chrono::high_resolution_clock::now() - start_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

int64_t Timer::GetMicroseconds() const {
    auto elapsed = std::chrono::high_resolution_clock::now() - start_;
    return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}

void Timer::Reset() {
    start_ = std::chrono::high_resolution_clock::now();
}
