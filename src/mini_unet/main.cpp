#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <timer/Timer.hpp>

torch::Tensor transform_image(const cv::Mat& image, int height, int width) {

    // Resize the image using OpenCV
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height), 0, 0, 0);
    // Convert back to tensor
    torch::Tensor out = torch::from_blob(resized_image.data, {1, 3, height, width}, torch::kInt8).contiguous();
    // torch::Tensor out = torch::from_blob(resized_image.data, {1, 3, height, width}, torch::kInt8).permute({0, 1, 2, 3}).contiguous();
    return out.to(torch::kFloat32).clone();
}

cv::Mat tensor_to_mat(const torch::Tensor& tensor) {
    // Ensure the tensor is on the CPU and in the correct type
    auto tensor_cpu = tensor.to(torch::kCPU).to(torch::kU8);
    
    // Get the tensor dimensions (C x H x W)
    int height = tensor_cpu.size(2);
    int width = tensor_cpu.size(3);

    // Create an OpenCV Mat with the correct size and type
    cv::Mat mat(cv::Size(width, height), CV_8UC1, tensor_cpu.data_ptr<uchar>());

    // OpenCV uses BGR format, so we need to reorder the channels if the tensor is RGB
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

    return mat;
}

void test_model(torch::jit::script::Module& model, std::string prefix_out_name) {
    cv::Mat image_10 = cv::imread("dataset/image/10.jpg");
    auto transformed_img = transform_image(image_10, 224, 224);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(transformed_img);
    auto output = model.forward(inputs).toTensor();
    auto output_img = tensor_to_mat(output);
    cv::imwrite(prefix_out_name + "_out_10.jpg", output_img);

    cv::Mat image_11 = cv::imread("dataset/image/11.jpg");
    auto transformed_img11 = transform_image(image_11, 224, 224);
    std::vector<torch::jit::IValue> inputs11;
    inputs11.push_back(transformed_img11);
    auto output11 = model.forward(inputs11).toTensor();
    auto output_img11 = tensor_to_mat(output11);
    cv::imwrite(prefix_out_name + "_out_11.jpg", output_img11);

    cv::Mat image_23 = cv::imread("dataset/image/23.jpg");
    auto transformed_img23 = transform_image(image_23, 224, 224);
    std::vector<torch::jit::IValue> inputs23;
    inputs23.push_back(transformed_img23);
    auto output23 = model.forward(inputs23).toTensor();
    auto output_img23 = tensor_to_mat(output23);
    cv::imwrite(prefix_out_name + "_out_23.jpg", output_img23);
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
