// #include "quadrilinear4d_kernel.h"
#include <torch/extension.h>
// #include <THC/THC.h>

int quadrilinear4d_forward_cuda(torch::Tensor luts, torch::Tensor tri_index, torch::Tensor weight, torch::Tensor image1, torch::Tensor image2, torch::Tensor image3, torch::Tensor image4, torch::Tensor output,
                           int lut_num, int lut_dim, int shift, float binsize, int width, int height, int batch);
// {
//     // Grab the input tensor
//     float * lut_flat = lut.data<float>();
//     float * image_flat = image.data<float>();
//     float * output_flat = output.data<float>();

//     QuadriLinearForwardLaucher(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, batch);

//     return 1;
// }

int quadrilinear4d_backward_cuda(torch::Tensor luts, torch::Tensor tri_index, torch::Tensor weight, torch::Tensor weight_grad, torch::Tensor image1, torch::Tensor image2, torch::Tensor image3, torch::Tensor image4, torch::Tensor output_grad,
                           int lut_num, int lut_dim, int shift, float binsize, int width, int height, int batch);
// {
//     // Grab the input tensor
//     float * image_grad_flat = image_grad.data<float>();
//     float * image_flat = image.data<float>();
//     float * lut_flat = lut.data<float>();
//     float * lut_grad_flat = lut_grad.data<float>();

//     QuadriLinearBackwardLaucher(image_flat, image_grad_flat, lut_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, batch);

//     return 1;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &quadrilinear4d_forward_cuda, "Quadrilinear forward");
  m.def("backward", &quadrilinear4d_backward_cuda, "Quadrilinear backward");
}

