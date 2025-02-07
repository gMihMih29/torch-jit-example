#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <timer/Timer.hpp>

void test_my_model() {
    Timer t;
    std::string scripted_model_path = "./models/scripted_MyModel.pt";
    std::string traced_model_path = "./models/traced_MyModel.pt";
    torch::jit::script::Module scripted_model, traced_model;
    // Scripted
    try {
        scripted_model = torch::jit::load(scripted_model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return;
    }
    std::cout << "scripted_model was loaded successfully!\n";
    torch::Tensor input_tensor = torch::randn({1, 10});
    std::cout << "Input tensor: " << input_tensor << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    t.Reset();
    auto output1 = scripted_model.forward(inputs).toTensor();
    auto time1 = t.GetMicroseconds();
    std::cout << "Output: " << output1 << std::endl << "Elaplsed: " << time1 << std::endl;

    std::cout << "\n\n\n";
    // Traced
    try {
        traced_model = torch::jit::load(traced_model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return;
    }
    std::cout << "traced_model was loaded successfully!\n";
    input_tensor = torch::randn({1, 10});
    std::cout << "Input tensor: " << input_tensor << std::endl;

    inputs.clear();
    inputs.push_back(input_tensor);
    t.Reset();
    auto output2 = traced_model.forward(inputs).toTensor();
    auto time2 = t.GetMicroseconds();
    std::cout << "Output: " << output2 << std::endl << "Elaplsed: " << time2 << std::endl;
}

void test_my_model_with_control_flow() {
    
}

int main() {
    test_my_model();
    test_my_model_with_control_flow();
    return 0;
}
