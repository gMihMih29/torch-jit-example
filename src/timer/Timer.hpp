#pragma once

#include <chrono>

class Timer {
public:
    Timer();

    int64_t GetMilliseconds() const;
    int64_t GetMicroseconds() const;
    void Reset();

private:
    std::chrono::_V2::system_clock::time_point start_;
};
