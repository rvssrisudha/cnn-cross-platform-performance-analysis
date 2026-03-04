#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

typedef short int16;
typedef int   int32;

// Dimensions based on source documentation [cite: 30, 41, 46, 49, 52]
#define IN_W 28
#define IN_H 28
#define C1_OUT 3
#define C1_K 5
#define C1_W 24
#define C1_H 24
#define MP1_W 12
#define MP1_H 12
#define C2_OUT 3
#define C2_K 5
#define C2_W 8
#define C2_H 8
#define MP2_W 4
#define MP2_H 4
#define FC_IN 48
#define FC_OUT 10

// Corrected Hex Loader: Handles 2's complement 8-bit hex to signed 16-bit
void load_mem_hex(const char *filename, std::vector<int16> &data) {
    std::ifstream file(filename);
    std::string token;
    if (!file.is_open()) { std::cerr << "Error opening " << filename << std::endl; return; }
    while(file >> token) {
        unsigned int raw = std::stoul(token, nullptr, 16);
        int8_t signed_8 = (int8_t)raw; // Critical for 2's complement interpretation
        data.push_back((int16)signed_8);
    }
}

void load_mem_dec(const char *filename, std::vector<int16> &data) {
    std::ifstream file(filename);
    int value;
    while(file >> value) data.push_back((int16)value);
}

// ================= KERNELS WITH 7-BIT FIXED POINT SCALING =================


__global__ void conv1_kernel(int16 *input, int16 *kernel, int16 *bias, int32 *output) {
    int oc = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= C1_W || y >= C1_H) return;

    // Align bias to product scale (input*weight) by shifting left 7 bits
    int32 sum = (int32)bias[oc] << 7;

    for(int i=0; i<C1_K; i++)
        for(int j=0; j<C1_K; j++)
            sum += (int32)input[(y+i)*IN_W + (x+j)] * (int32)kernel[oc*C1_K*C1_K + i*C1_K + j];

    // Quantization: Shift right by 7 to normalize back for ReLU/Pooling
    output[oc*C1_W*C1_H + y*C1_W + x] = sum >> 7;
}

__global__ void conv2_kernel(int32 *input, int16 *kernel, int16 *bias, int32 *output) {
    int oc = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= C2_W || y >= C2_H) return;

    int32 sum = (int32)bias[oc] << 7;

    for(int ic=0; ic<3; ic++) {
        for(int i=0; i<C2_K; i++) {
            for(int j=0; j<C2_K; j++) {
                int32 in_val = input[ic*MP1_W*MP1_H + (y+i)*MP1_W + (x+j)];
                int16 w = kernel[oc*3*C2_K*C2_K + ic*C2_K*C2_K + i*C2_K + j];
                sum += in_val * w;
            }
        }
    }
    output[oc*C2_W*C2_H + y*C2_W + x] = sum >> 7;
}

__global__ void relu_kernel(int32 *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size && data[idx] < 0) data[idx] = 0; // ReLU non-linearity [cite: 8, 235]
}


__global__ void maxpool_kernel(int32 *input, int32 *output, int channels, int in_w, int in_h, int out_w, int out_h) {
    int c = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= out_w || y >= out_h) return;

    int32 a = input[c*in_w*in_h + (2*y)*in_w + (2*x)];
    int32 b = input[c*in_w*in_h + (2*y)*in_w + (2*x+1)];
    int32 c1 = input[c*in_w*in_h + (2*y+1)*in_w + (2*x)];
    int32 d = input[c*in_w*in_h + (2*y+1)*in_w + (2*x+1)];

    output[c*out_w*out_h + y*out_w + x] = max(max(a,b), max(c1,d)); // Downsampling [cite: 8, 16]
}

__global__ void fc_kernel(int32 *input, int16 *weights, int16 *bias, int32 *output) {
    int neuron = threadIdx.x;
    if(neuron >= FC_OUT) return;

    int32 sum = (int32)bias[neuron] << 7;

    // Flattening: Ensure CHW order to match PyTorch's view()
    for(int i=0; i<FC_IN; i++)
        sum += input[i] * weights[neuron*FC_IN + i];

    output[neuron] = sum >> 7;
}

// ================= MAIN EXECUTION =================

int main() {
    // 1. Load Input Image (Decimal as per Python export) [cite: 98]
    std::vector<int16> input_img;
    load_mem_dec("test_image.mem", input_img);

    // 2. Load Weights/Biases (HEX as per notebook export) [cite: 103, 104]
    std::vector<int16> c1k, c1b, c2k, c2b, fcw, fcb;
    std::vector<int16> temp;

    load_mem_hex("conv1_weight_1.mem", temp); c1k.insert(c1k.end(), temp.begin(), temp.end());
    load_mem_hex("conv1_weight_2.mem", temp); c1k.insert(c1k.end(), temp.begin(), temp.end());
    load_mem_hex("conv1_weight_3.mem", temp); c1k.insert(c1k.end(), temp.begin(), temp.end());
    load_mem_hex("conv1_bias.mem", c1b);

    for(int oc=1; oc<=3; oc++) {
        for(int ic=1; ic<=3; ic++) {
            std::string fname = "conv2_weight_" + std::to_string(oc) + std::to_string(ic) + ".mem";
            temp.clear(); load_mem_hex(fname.c_str(), temp);
            c2k.insert(c2k.end(), temp.begin(), temp.end());
        }
    }
    load_mem_hex("conv2_bias.mem", c2b);
    load_mem_hex("fc_weight.mem", fcw);
    load_mem_hex("fc_bias.mem", fcb);

    // 3. Device Memory Setup
    int16 *d_in, *d_c1k, *d_c1b, *d_c2k, *d_c2b, *d_fcw, *d_fcb;
    int32 *d_c1o, *d_mp1, *d_c2o, *d_mp2, *d_fco;

    cudaMalloc(&d_in, input_img.size()*2);
    cudaMalloc(&d_c1k, c1k.size()*2); cudaMalloc(&d_c1b, c1b.size()*2);
    cudaMalloc(&d_c2k, c2k.size()*2); cudaMalloc(&d_c2b, c2b.size()*2);
    cudaMalloc(&d_fcw, fcw.size()*2); cudaMalloc(&d_fcb, fcb.size()*2);
    cudaMalloc(&d_c1o, C1_OUT*C1_W*C1_H*4);
    cudaMalloc(&d_mp1, C1_OUT*MP1_W*MP1_H*4);
    cudaMalloc(&d_c2o, C2_OUT*C2_W*C2_H*4);
    cudaMalloc(&d_mp2, C2_OUT*MP2_W*MP2_H*4);
    cudaMalloc(&d_fco, FC_OUT*4);

    cudaMemcpy(d_in, input_img.data(), input_img.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1k, c1k.data(), c1k.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1b, c1b.data(), c1b.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2k, c2k.data(), c2k.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2b, c2b.data(), c2b.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcw, fcw.data(), fcw.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcb, fcb.data(), fcb.size()*2, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // 4. CNN Forward Flow [cite: 65-70, 76]
    dim3 block(16, 16);

    // Layer 1
    conv1_kernel<<<dim3(2,2,3), block>>>(d_in, d_c1k, d_c1b, d_c1o);
    relu_kernel<<<(C1_OUT*C1_W*C1_H+255)/256, 256>>>(d_c1o, C1_OUT*C1_W*C1_H);
    maxpool_kernel<<<dim3(1,1,3), block>>>(d_c1o, d_mp1, 3, C1_W, C1_H, MP1_W, MP1_H);

    // Layer 2
    conv2_kernel<<<dim3(1,1,3), block>>>(d_mp1, d_c2k, d_c2b, d_c2o);
    relu_kernel<<<(C2_OUT*C2_W*C2_H+255)/256, 256>>>(d_c2o, C2_OUT*C2_W*C2_H);
    maxpool_kernel<<<dim3(1,1,3), block>>>(d_c2o, d_mp2, 3, C2_W, C2_H, MP2_W, MP2_H);

    // FC Layer
    fc_kernel<<<1, FC_OUT>>>(d_mp2, d_fcw, d_fcb, d_fco);

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 5. Result Verification [cite: 313-315]
    std::vector<int32> final_out(FC_OUT);
    cudaMemcpy(final_out.data(), d_fco, FC_OUT*4, cudaMemcpyDeviceToHost);

    int prediction = 0;
    for(int i=1; i<FC_OUT; i++) {
        if(final_out[i] > final_out[prediction]) prediction = i;
    }

    std::cout << "Predicted Digit: " << prediction << std::endl;
    std::cout << "GPU execution time for prediction: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(d_in); cudaFree(d_c1k); cudaFree(d_c1b);
    cudaFree(d_c2k); cudaFree(d_c2b); cudaFree(d_fcw); cudaFree(d_fcb);
    cudaFree(d_c1o); cudaFree(d_mp1); cudaFree(d_c2o); cudaFree(d_mp2); cudaFree(d_fco);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
