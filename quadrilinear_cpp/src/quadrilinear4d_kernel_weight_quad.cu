#include <math.h>
#include <float.h>
// #include "quadrilinear4d_kernel.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
#include <assert.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


__global__ void QuadriLinearForward(const int nthreads, const float* luts, const float* weight, const float* image1, const float* image2, const float* image3, const float* image4, float* output, const int luts_num, const int dim, const int shift, const float binsize, const int width, const int height, const int batch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
        
        int index_batch = floor(index / (width * height));
        int index_height = floor((index - (index_batch * width * height)) / width);
        int index_width = index - index_batch * width * height - index_height * width;
        // index = index_batch * width * height + index_height * width + index_width
        for (int index_channel = 0; index_channel < 3; index_channel += 1)
        {
            float a = image1[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            float b = image2[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            float c = image3[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            float d = image4[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            
            int a_id = floor(a / binsize);
            int b_id = floor(b / binsize);
            int c_id = floor(c / binsize);
            int d_id = floor(d / binsize);

            float a_d = fmod(a,binsize) / binsize;
            float b_d = fmod(b,binsize) / binsize;
            float c_d = fmod(c,binsize) / binsize;
            float d_d = fmod(d,binsize) / binsize;

            int id0000 = (a_id * dim * dim * dim + b_id * dim * dim + c_id * dim + d_id)*shift*shift;
            int id0100 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + d_id)*shift*shift;
            int id0010 = (a_id * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id0001 = (a_id * dim * dim * dim + b_id * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id0110 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id0011 = (a_id * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;
            int id0101 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id0111 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;

            int id1000 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + c_id * dim + d_id)*shift*shift;
            int id1100 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + d_id)*shift*shift;
            int id1010 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id1001 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id1110 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id1011 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;
            int id1101 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id1111 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;

            float w0000 = (1-a_d)*(1-b_d)*(1-c_d)*(1-d_d);
            float w0100 = (1-a_d)*b_d*(1-c_d)*(1-d_d);
            float w0010 = (1-a_d)*(1-b_d)*c_d*(1-d_d);
            float w0001 = (1-a_d)*(1-b_d)*(1-c_d)*d_d;
            float w0110 = (1-a_d)*b_d*c_d*(1-d_d);
            float w0011 = (1-a_d)*(1-b_d)*c_d*d_d;
            float w0101 = (1-a_d)*b_d*(1-c_d)*d_d;
            float w0111 = (1-a_d)*b_d*c_d*d_d;

            float w1000 = a_d*(1-b_d)*(1-c_d)*(1-d_d);
            float w1100 = a_d*b_d*(1-c_d)*(1-d_d);
            float w1010 = a_d*(1-b_d)*c_d*(1-d_d);
            float w1001 = a_d*(1-b_d)*(1-c_d)*d_d;
            float w1110 = a_d*b_d*c_d*(1-d_d);
            float w1011 = a_d*(1-b_d)*c_d*d_d;
            float w1101 = a_d*b_d*(1-c_d)*d_d;
            float w1111 = a_d*b_d*c_d*d_d;

            // 4x4 output pixel
            int lut_step = dim * dim * dim * dim * shift * shift;
            int output_index = (index_batch*3*height*width + index_channel*height*width + index_height*width + index_width)*shift*shift;
            // for each LUT base 
            for (int j = 0; j < luts_num; j += 1)
                // for 4x4 pixel
                for (int i = 0; i < shift*shift; i += 1)
                {   
                    float w = weight[index_batch*luts_num*height*width + j*height*width + index_height*width + index_width];
                    output[output_index+i] += w*(w0000 * luts[j*lut_step+id0000+i] + w0100 * luts[j*lut_step+id0100+i] + w0010 * luts[j*lut_step+id0010+i] + 
                                    w0001 * luts[j*lut_step+id0001+i] + w0110 * luts[j*lut_step+id0110+i] + w0011 * luts[j*lut_step+id0011+i] + 
                                    w0101 * luts[j*lut_step+id0101+i] + w0111 * luts[j*lut_step+id0111+i] +
                                    w1000 * luts[j*lut_step+id1000+i] + w1100 * luts[j*lut_step+id1100+i] + w1010 * luts[j*lut_step+id1010+i] + 
                                    w1001 * luts[j*lut_step+id1001+i] + w1110 * luts[j*lut_step+id1110+i] + w1011 * luts[j*lut_step+id1011+i] + 
                                    w1101 * luts[j*lut_step+id1101+i] + w1111 * luts[j*lut_step+id1111+i]);
                }
        }
    }
}


int QuadriLinearForwardLaucher(const float* luts, const float* weight, const float* image1, const float* image2, const float* image3, const float* image4, float* output, const int luts_num, const int luts_dim, const int shift, const float binsize, const int width, const int height, const int batch) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;


    QuadriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0>>>(output_size, luts, weight, image1, image2, image3, image4, output, luts_num,  luts_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

int quadrilinear4d_forward_cuda(torch::Tensor luts, torch::Tensor weight, torch::Tensor image1, torch::Tensor image2, torch::Tensor image3, torch::Tensor image4, torch::Tensor output,
                           int luts_num, int luts_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * luts_flat = luts.data<float>();
    float * weight_flat = weight.data<float>();
    float * image_flat1 = image1.data<float>();
    float * image_flat2 = image2.data<float>();
    float * image_flat3 = image3.data<float>();
    float * image_flat4 = image4.data<float>();
    float * output_flat = output.data<float>();

    QuadriLinearForwardLaucher(luts_flat, weight_flat, image_flat1, image_flat2, image_flat3, image_flat4, output_flat, luts_num, luts_dim, shift, binsize, width, height, batch);

    return 1;
}

