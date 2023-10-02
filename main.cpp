#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "HalideBuffer.h"
#include "adjust_brightness.h"
#include "rgb_to_grayscale.h"
#include "invert.h"
#include "hflip.h"
#include "vflip.h"
#include "erase.h"
#include "erase_2.h"
#include "solarize.h"
#include "autocontrast.h"
#include "crop.h"
#include "adjust_gamma.h"
#include "adjust_saturation.h"
#include "adjust_contrast.h"
#include "normalize.h"
#include "posterize.h"
#include "adjust_hue.h"
#include "adjust_sharpness.h"
#include "elastic_transform.h"


std::vector<int> get_dims(const torch::Tensor &tensor) { // Helper for wrap()
    int ndims = tensor.ndimension();
    std::vector<int> dims(ndims, 0);
    // PyTorch dim order is reverse of Halide
    for (int dim = 0; dim < ndims; ++dim) {
        dims[dim] = tensor.size(ndims - 1 - dim);
    }
    return dims;
}

// Converts Buffer to a 3D matrix (C*H*W)
torch::Tensor ConvertBufferToTensor(const Halide::Runtime::Buffer<float> &buffer) {
    torch::Tensor tensor = torch::from_blob(buffer.data(), {buffer.channels(), buffer.height(), buffer.width()}, torch::TensorOptions().dtype(torch::kFloat32));
    return tensor;
}

Halide::Runtime::Buffer<float> ConvertTensorToBuffer(const torch::Tensor &tensor) { // Function to wrap a tensor in a Halide Buffer
    std::vector<int> dims = get_dims(tensor);
    float* pData = tensor.data_ptr<float>();
    return Halide::Runtime::Buffer<float>(pData, dims);
}

Halide::Runtime::Buffer<float> CloneDims(const Halide::Runtime::Buffer<float> &input) {
    return Halide::Runtime::Buffer<float>(input.width(), input.height(), input.channels());
}

torch::Tensor ConvertBufferToTensor_int(const Halide::Runtime::Buffer<uint8_t> &buffer) {
    // Create a PyTorch tensor using the from_blob method
    torch::Tensor tensor = torch::from_blob(buffer.data(), {buffer.channels(), buffer.height(), buffer.width()},
                                            torch::TensorOptions().dtype(torch::kUInt8));
    return tensor;
}

Halide::Runtime::Buffer<uint8_t> ConvertTensorToBuffer_int(torch::Tensor &tensor) { // Function to wrap an ATen tensor in a Halide Buffer
    std::vector<int> dims = get_dims(tensor);
    uint8_t* pData = tensor.data_ptr<uint8_t>();
    return Halide::Runtime::Buffer<uint8_t>(pData, dims);
}


void adjust_brightness(torch::Tensor &tensor, float factor, torch::Tensor &tensor_out) {
    if(factor < 0) {
        throw std::invalid_argument("Brightness factor must be >=0\n");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_brighter(input, factor, output);
}

void grayscale(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_grayscale(input, output);
}

void invert(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_invert(input, output);
}

void invert_batch(std::vector<torch::Tensor> in, std::vector<torch::Tensor> out) {
    if(in.size() != out.size()) {
        throw std::invalid_argument("size of input and output lists are not equal\n");
    }
    for (int i = 0; i < in.size(); i++) {
        Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(in[i]);
        Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(out[i]);
        halide_invert(input, output);
    }

    return;
}

void hflip(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_hflip(input, output);
}

void vflip(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_vflip(input, output);
}


//  TODO: figure out pybind optional arguments to make these 2 functions into 1
void erase_tensor(const torch::Tensor &tensor, int i, int j, int h, int w, torch::Tensor &v, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> other = ConvertTensorToBuffer(v);
    if (i + w > input.width() || j + h > input.height()) {  // Ensure erase area is in bounds
        throw std::invalid_argument("Dimensions for erasure are out of bounds\n");
    } else if (w != other.width() || h != other.height()) {  // Ensure that new tensor fits the dims
        throw std::invalid_argument("Dimensions for other tensor do not match height/width\n");
    }

    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    halide_erase(input, i, j, w, h, other, output);
}

void erase(const torch::Tensor &tensor, int i, int j, int h, int w, float v, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    if (i + w > input.width() || j + h > input.height()) {  // Ensure erase area is in bounds
        throw std::invalid_argument("Dimensions for erasure are out of bounds\n");
    }
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    halide_erase_2(input, i, j, w, h, v, output);
}

void crop(torch::Tensor &tensor, int i, int j, int h, int w, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output(tensor_out.data_ptr<float>(), {w, h, 3});

    halide_crop(input, i, j, h, w, output);
}

void solarize(torch::Tensor &tensor, float threshold, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_solarize(input, threshold, output);
}

void elastic_transform(torch::Tensor &tensor, torch::Tensor &dis, torch::Tensor &tensor_out, float fill=0){
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> displacement = ConvertTensorToBuffer(dis);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    if (displacement.width()!=2 || displacement.height()!=input.width() || displacement.channels()!=input.height() || displacement.dim(3).extent()!=1){
        throw std::invalid_argument("The displacement expected shape is [1,H,W,2], the displacement passed has a invalid shape.");
    }

    halide_elastic_transform(input, displacement, fill, output);

}

void autocontrast(torch::Tensor &tensor, torch::Tensor &tensor_out){
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    float min_r = 1;
    float min_g = 1;
    float min_b = 1;
    float max_r = 0;
    float max_g = 0;
    float max_b = 0;
    float element;
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            element = input(x,y,0);
            if (element < min_r) min_r = element;
            if (element > max_r) max_r = element;
        }
    }
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            element = input(x,y,1);
            if (element < min_g) min_g = element;
            if (element > max_g) max_g = element;
        }
    }
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            element = input(x,y,2);
            if (element < min_b) min_b = element;
            if (element > max_b) max_b = element;
        }
    }


	//if any element of 1/(maximum - minimum) is infinite, that element in minimum and scale will be 0 and 1.
    float scale_r;
    float scale_g;
    float scale_b;
    if (max_r - min_r == 0) {
        scale_r = 1;
        min_r = 0;
    } else {
        scale_r = 1/(max_r - min_r);
    }
    if (max_g - min_g == 0) {
        scale_g = 1;
        min_g = 0;
    } else {
        scale_g = 1/(max_g - min_g);
    }
    if (max_b - min_b == 0) {
        scale_b = 1;
        min_b = 0;
    } else {
        scale_b = 1/(max_b - min_b);
    }
    halide_autocontrast(input, min_r, min_g, min_b, scale_r, scale_g, scale_b, output);
}

void adjust_gamma(torch::Tensor &tensor, float gamma, float gain, torch::Tensor &tensor_out) {

    if(gamma < 0) {
        throw std::invalid_argument("gamma must be non-negative");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_adjust_gamma(input, gamma, gain, output);
}

void adjust_saturation(torch::Tensor &tensor, float saturation, torch::Tensor &tensor_out) {
    if(saturation < 0) {
        throw std::invalid_argument("saturation factor must be non-negative");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_adjust_saturation(input, saturation, output);
}

void adjust_contrast(torch::Tensor &tensor, float contrast, torch::Tensor &tensor_out) {
    if(contrast < 0) {
        throw std::invalid_argument("contrast factor must be non-negative");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    Halide::Runtime::Buffer<float> input2 = ConvertTensorToBuffer(tensor);
    Halide::Runtime::Buffer<float> gray(input.width(), input.height());
    halide_grayscale(input2, gray);
    float total = 0.0;
    for(int j = 0; j < input.height(); j++) {
        for(int k = 0; k < input.width(); k++) {
            total += gray(k, j);
        }
    }
    halide_adjust_contrast(input, contrast, total/(input.width() * input.height()), output);
}

void normalize(torch::Tensor &tensor, std::vector<float> mean, std::vector<float> sd, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    // This is supposed to add support for 2 channel images, does not work since halide_normalize will always expect
    // 2 more args than are provided if channels != 3
    if(input.channels() == 3) {
        halide_normalize(input, mean[0], mean[1], mean[2], sd[0], sd[1], sd[2], output);
    } else {
        //halide_normalize(input, mean[0], mean[1], sd[0], sd[1]);
    }
}

void posterize(torch::Tensor &tensor, int bits, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<uint8_t> input = ConvertTensorToBuffer_int(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<uint8_t> output = ConvertTensorToBuffer_int(tensor_out);
    if(input.channels() != 3) {
        throw std::invalid_argument("Posterize required a tensor with 3 dimensions");
    }
    int8_t mask = std::pow(2, (8-bits)) * -1;
    halide_posterize(input, mask, output);
}

void adjust_hue(torch::Tensor &tensor, float hue_factor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_adjust_hue(input, hue_factor, output);
}


void adjust_sharpness(torch::Tensor &tensor, float sharpness_factor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_adjust_sharpness(input, sharpness_factor, output);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adjust_brightness", &adjust_brightness);
    m.def("rgb_to_grayscale", &grayscale);
    m.def("invert", &invert);
    m.def("invert_batch", &invert_batch);
    m.def("hflip", &hflip);
    m.def("vflip", &vflip);
    m.def("erase_tensor", &erase_tensor);
    m.def("erase", &erase);
    m.def("solarize", &solarize);
    m.def("autocontrast", &autocontrast);
    m.def("crop", &crop);
    m.def("adjust_gamma", &adjust_gamma);
    m.def("adjust_saturation", &adjust_saturation);
    m.def("adjust_contrast", &adjust_contrast);
    m.def("normalize", &normalize);
    m.def("posterize", &posterize);
    m.def("adjust_hue", &adjust_hue);
    m.def("adjust_sharpness", &adjust_sharpness);
    m.def("elastic_transform", &elastic_transform);
}
