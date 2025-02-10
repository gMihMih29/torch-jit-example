#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <timer/Timer.hpp>

torch::Tensor transform_image(const cv::Mat& image, int height, int width) {

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height), 0, 0, 0);
    torch::Tensor out = torch::from_blob(resized_image.data, {1, 3, height, width}, torch::kInt8).contiguous();
    // torch::Tensor out = torch::from_blob(resized_image.data, {1, 3, height, width}, torch::kInt8).permute({0, 1, 2, 3}).contiguous();
    return (out.to(torch::kFloat32).clone() + 128) / 255.f;
}

cv::Mat tensor_to_mat(const torch::Tensor& tensor) {
    auto tensor_cpu = (tensor * 255.0f).to(torch::kUInt8);

    int height = tensor_cpu.size(2);
    int width = tensor_cpu.size(3);

    cv::Mat mat(cv::Size(width, height), CV_8UC1, tensor_cpu.data_ptr<uchar>());

    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

    return mat;
}

void test_case(torch::jit::script::Module& model, std::string img_path, std::string out_name, float threshold = 0.5f) {
    cv::Mat image_10 = cv::imread(img_path);
    auto transformed_img = transform_image(image_10, 224, 224);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(transformed_img);

    auto output = model.forward(inputs).toTensor();
    output = ((output / torch::max(output).item()) > threshold);
    auto output_img = tensor_to_mat(output);
    cv::imwrite(out_name, output_img);
}

void test_model(torch::jit::script::Module& model, std::string prefix_out_name) {

    test_case(model, "dataset/image/10.jpg", prefix_out_name + "_out_10.jpg");
    test_case(model, "dataset/image/11.jpg", prefix_out_name + "_out_11.jpg");
    test_case(model, "dataset/image/23.jpg", prefix_out_name + "_out_23.jpg");

    // cv::Mat image_10 = cv::imread("dataset/image/10.jpg");
    // auto transformed_img = transform_image(image_10, 224, 224);
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(transformed_img);
    // auto output = model.forward(inputs).toTensor();
    // output = ((output / torch::max(output).item()) > threshold);
    // auto output_img = tensor_to_mat(output);
    // cv::imwrite(prefix_out_name + "_out_10.jpg", output_img);

    // cv::Mat image_11 = cv::imread("dataset/image/11.jpg");
    // auto transformed_img11 = transform_image(image_11, 224, 224);
    // std::vector<torch::jit::IValue> inputs11;
    // inputs11.push_back(transformed_img11);
    // auto output11 = model.forward(inputs11).toTensor();
    // // output11 = ((output11 / torch::max(output11).item()) > threshold).to(torch::kFloat32);
    // output11 = ((output11 / torch::max(output11).item()) > threshold);
    // auto output_img11 = tensor_to_mat(output11);
    // cv::imwrite(prefix_out_name + "_out_11.jpg", output_img11);

    // cv::Mat image_23 = cv::imread("dataset/image/23.jpg");
    // auto transformed_img23 = transform_image(image_23, 224, 224);
    // std::vector<torch::jit::IValue> inputs23;
    // inputs23.push_back(transformed_img23);
    // auto output23 = model.forward(inputs23).toTensor();
    // output23 = ((output23 / torch::max(output23).item()) > threshold);
    // auto output_img23 = tensor_to_mat(output23);
    // cv::imwrite(prefix_out_name + "_out_23.jpg", output_img23);
}

int main() {
    std::string scripted_model_path = "./models/scripted_mini_unet.pt";
    std::string traced_model_path = "./models/traced_mini_unet.pt";
    torch::jit::script::Module scripted_model, traced_model;
    // Scripted
    try {
        scripted_model = torch::jit::load(scripted_model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 0;
    }
    test_model(scripted_model, "scripted");
    std::cout << "\n\n\n";
    try {
        traced_model = torch::jit::load(traced_model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 0;
    }
    //
    test_model(traced_model, "traced");
    return 0;
}
