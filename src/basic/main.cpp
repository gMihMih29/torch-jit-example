#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <timer/Timer.hpp>

void predict(torch::jit::script::Module& model, torch::Tensor input) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    Timer t;
    auto output1 = model.forward(inputs).toTensor();
    std::cout << "Output: " << output1 << std::endl << "Elaplsed: " << t.GetMicroseconds() << std::endl;
}

void test_my_model() {
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
    torch::Tensor input_tensor = torch::randn({1, 10});
    std::cout << "Input tensor: " << input_tensor << std::endl;
    std::cout << "Scripted:\n";
    predict(scripted_model, input_tensor);

    std::cout << "\n\n\n";
    // Traced
    try {
        traced_model = torch::jit::load(traced_model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return;
    }
    input_tensor = torch::randn({1, 10});
    std::cout << "Input tensor: " << input_tensor << std::endl;

    std::cout << "Traced:\n";
    predict(traced_model, input_tensor);

    std::cout << '\n';
    // Invalid input size:
    std::cout << "Invalid input size:\n";
    torch::Tensor in1 = torch::randn({1, 11});
    std::cout << "Input tensor: " << in1 << std::endl;

    predict(traced_model, in1);
}

void test_my_model_with_control_flow() {
    std::cout << "\n\n\n";
    Timer t;
    // Scripted with control flow:
    // def forward(self,
    //     x: Tensor) -> Tensor:
    // if bool(torch.gt(torch.sum(x), 0)):
    //     _0 = x
    // else:
    //     _0 = torch.neg(x)
    // return _0
    std::string scripted_model_path = "./models/scripted_MyModelWithControlFlow.pt";

    // Traced with control flow:
    // def forward(self,
    //     x: Tensor) -> Tensor:
    // return torch.neg(x)
    std::string traced_model_path = "./models/traced_MyModelWithControlFlow.pt";
    torch::jit::script::Module scripted_model, traced_model;
    // Scripted
    try {
        scripted_model = torch::jit::load(scripted_model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return;
    }
    std::cout << "Scripted model with control flow:\n";
    torch::Tensor input_tensor = torch::tensor({1});
    std::cout << "Input tensor: " << input_tensor << std::endl;

    predict(scripted_model, input_tensor);
    std::cout << "Expected: 1\n";

    std::cout << "\n";

    input_tensor = torch::tensor({-1});
    std::cout << "Input tensor: " << input_tensor << std::endl;

    predict(scripted_model, input_tensor);
    std::cout << "Expected: 1\n";

    std::cout << "\n\n\n";
    // Traced
    try {
        traced_model = torch::jit::load(traced_model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return;
    }

    std::cout << "Traced model with control flow:\n";
    input_tensor = torch::tensor({-1});
    std::cout << "Input tensor: " << input_tensor << std::endl;
    predict(traced_model, input_tensor);
    std::cout << "Expected: 1\n";

    std::cout << "\n";
    input_tensor = torch::tensor({1});
    std::cout << "Input tensor: " << input_tensor << std::endl;
    predict(traced_model, input_tensor);
    std::cout << "Expected: 1\n";
}

int main() {
    test_my_model();
    test_my_model_with_control_flow();
    return 0;
}
