#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <timer/Timer.hpp>

int main() {
    std::string func_path = "./models/my_func.pt";
    torch::jit::script::Module func;
    try {
        func = torch::jit::load(func_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the func: " << e.what() << std::endl;
        return 0;
    }
    torch::Tensor input = torch::tensor({1, 2, 3, 4, 5});
    std::cout << func.forward({input}) << "\n";
    input = torch::tensor({1, 2, 3, -4, -5});
    std::cout << func.forward({input}) << "\n";
    return 0;
}
