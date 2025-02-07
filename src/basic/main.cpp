#include <torch/script.h>
#include <torch/torch.h>
#include <opencv4/opencv2/opencv.hpp>

#include <iostream>

int main() {
    std::string model_path = "model.pt";
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        std::cout << "Model was loaded successful!" << std::endl;

        std::vector<float> input_data(10, 1.0);
        torch::Tensor input_tensor = torch::from_blob(input_data.data(), {1, 10});

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        at::Tensor output = model.forward(inputs).toTensor();

        std::cout << "Model output: " << output << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
