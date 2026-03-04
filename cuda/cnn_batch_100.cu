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
#define BATCH_SIZE 100

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
    int oc = blockIdx.z % C1_OUT; // Output channel
    int batch_idx = blockIdx.z / C1_OUT; // Batch index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= C1_W || y >= C1_H || batch_idx >= BATCH_SIZE) return;

    // Calculate offset for the current batch image
    int input_batch_offset = batch_idx * IN_W * IN_H;
    int output_batch_offset = batch_idx * C1_OUT * C1_W * C1_H;

    // Align bias to product scale (input*weight) by shifting left 7 bits
    int32 sum = (int32)bias[oc] << 7;

    for(int i=0; i<C1_K; i++)
        for(int j=0; j<C1_K; j++)
            sum += (int32)input[input_batch_offset + (y+i)*IN_W + (x+j)] * (int32)kernel[oc*C1_K*C1_K + i*C1_K + j];

    // Quantization: Shift right by 7 to normalize back for ReLU/Pooling
    output[output_batch_offset + oc*C1_W*C1_H + y*C1_W + x] = sum >> 7;
}

__global__ void conv2_kernel(int32 *input, int16 *kernel, int16 *bias, int32 *output) {
    int oc = blockIdx.z % C2_OUT; // Output channel
    int batch_idx = blockIdx.z / C2_OUT; // Batch index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= C2_W || y >= C2_H || batch_idx >= BATCH_SIZE) return;

    // Calculate offset for the current batch image
    int input_batch_offset = batch_idx * C1_OUT * MP1_W * MP1_H; // C1_OUT here as it's from the previous layer
    int output_batch_offset = batch_idx * C2_OUT * C2_W * C2_H;

    int32 sum = (int32)bias[oc] << 7;

    for(int ic=0; ic<C1_OUT; ic++) { // Iterate through input channels, which is C1_OUT
        for(int i=0; i<C2_K; i++) {
            for(int j=0; j<C2_K; j++) {
                int32 in_val = input[input_batch_offset + ic*MP1_W*MP1_H + (y+i)*MP1_W + (x+j)];
                int16 w = kernel[oc*C1_OUT*C2_K*C2_K + ic*C2_K*C2_K + i*C2_K + j];
                sum += in_val * w;
            }
        }
    }
    output[output_batch_offset + oc*C2_W*C2_H + y*C2_W + x] = sum >> 7;
}

__global__ void relu_kernel(int32 *data, int size_per_image) {
    int batch_idx = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(batch_idx >= BATCH_SIZE) return;

    int offset = batch_idx * size_per_image;
    if(idx < size_per_image && data[offset + idx] < 0) data[offset + idx] = 0; // ReLU non-linearity [cite: 8, 235]
}


__global__ void maxpool_kernel(int32 *input, int32 *output, int channels, int in_w, int in_h, int out_w, int out_h) {
    int c = blockIdx.z % channels; // Channel index
    int batch_idx = blockIdx.z / channels; // Batch index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= out_w || y >= out_h || batch_idx >= BATCH_SIZE) return;

    int input_batch_offset = batch_idx * channels * in_w * in_h;
    int output_batch_offset = batch_idx * channels * out_w * out_h;

    int32 a = input[input_batch_offset + c*in_w*in_h + (2*y)*in_w + (2*x)];
    int32 b = input[input_batch_offset + c*in_w*in_h + (2*y)*in_w + (2*x+1)];
    int32 c1 = input[input_batch_offset + c*in_w*in_h + (2*y+1)*in_w + (2*x)];
    int32 d = input[input_batch_offset + c*in_w*in_h + (2*y+1)*in_w + (2*x+1)];

    output[output_batch_offset + c*out_w*out_h + y*out_w + x] = max(max(a,b), max(c1,d)); // Downsampling [cite: 8, 16]
}

__global__ void fc_kernel(int32 *input_batch, int16 *weights, int16 *bias, int32 *output_batch) {
    int batch_idx = blockIdx.x; // Each block processes one image in the batch
    int neuron = threadIdx.x; // Each thread processes one output neuron for that image

    if(batch_idx >= BATCH_SIZE || neuron >= FC_OUT) return;

    // Calculate offset for the current batch image's input and output
    int input_offset = batch_idx * FC_IN; // FC_IN is the flattened size of the input to FC layer
    int output_offset = batch_idx * FC_OUT;

    int32 sum = (int32)bias[neuron] << 7;

    for(int i=0; i<FC_IN; i++)
        sum += input_batch[input_offset + i] * weights[neuron*FC_IN + i];

    output_batch[output_offset + neuron] = sum >> 7;
}

// ================= MAIN EXECUTION =================

int main() {
    // 1. Load Input Image (Decimal as per Python export) [cite: 98]
    std::vector<int16> input_img;
    load_mem_dec("test_images_batch.mem", input_img);

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

    // Allocate d_in for BATCH_SIZE images
    cudaMalloc(&d_in, input_img.size() * sizeof(int16)); // input_img.size() already accounts for BATCH_SIZE

    cudaMalloc(&d_c1k, c1k.size()*sizeof(int16)); cudaMalloc(&d_c1b, c1b.size()*sizeof(int16));
    cudaMalloc(&d_c2k, c2k.size()*sizeof(int16)); cudaMalloc(&d_c2b, c2b.size()*sizeof(int16));
    cudaMalloc(&d_fcw, fcw.size()*sizeof(int16)); cudaMalloc(&d_fcb, fcb.size()*sizeof(int16));

    // Allocate output buffers to accommodate BATCH_SIZE
    cudaMalloc(&d_c1o, BATCH_SIZE * C1_OUT * C1_W * C1_H * sizeof(int32));
    cudaMalloc(&d_mp1, BATCH_SIZE * C1_OUT * MP1_W * MP1_H * sizeof(int32));
    cudaMalloc(&d_c2o, BATCH_SIZE * C2_OUT * C2_W * C2_H * sizeof(int32));
    cudaMalloc(&d_mp2, BATCH_SIZE * C2_OUT * MP2_W * MP2_H * sizeof(int32));
    cudaMalloc(&d_fco, BATCH_SIZE * FC_OUT * sizeof(int32));

    cudaMemcpy(d_in, input_img.data(), input_img.size()*sizeof(int16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1k, c1k.data(), c1k.size()*sizeof(int16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1b, c1b.data(), c1b.size()*sizeof(int16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2k, c2k.data(), c2k.size()*sizeof(int16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2b, c2b.data(), c2b.size()*sizeof(int16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcw, fcw.data(), fcw.size()*sizeof(int16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcb, fcb.data(), fcb.size()*sizeof(int16), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // 4. CNN Forward Flow [cite: 65-70, 76]
    const int THREADS_PER_BLOCK_2D = 16;
    const int THREADS_PER_BLOCK_1D = 256;

    // Layer 1
    dim3 block_2d(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);

    dim3 conv1_grid_dim(
        (C1_W + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        (C1_H + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        C1_OUT * BATCH_SIZE
    );
    conv1_kernel<<<conv1_grid_dim, block_2d>>>(d_in, d_c1k, d_c1b, d_c1o);

    dim3 relu1_grid_dim(
        (C1_OUT*C1_W*C1_H + THREADS_PER_BLOCK_1D - 1) / THREADS_PER_BLOCK_1D,
        1,
        BATCH_SIZE
    );
    relu_kernel<<<relu1_grid_dim, THREADS_PER_BLOCK_1D>>>(d_c1o, C1_OUT*C1_W*C1_H);

    dim3 mp1_grid_dim(
        (MP1_W + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        (MP1_H + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        C1_OUT * BATCH_SIZE
    );
    maxpool_kernel<<<mp1_grid_dim, block_2d>>>(d_c1o, d_mp1, C1_OUT, C1_W, C1_H, MP1_W, MP1_H);

    // Layer 2
    dim3 conv2_grid_dim(
        (C2_W + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        (C2_H + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        C2_OUT * BATCH_SIZE
    );
    conv2_kernel<<<conv2_grid_dim, block_2d>>>(d_mp1, d_c2k, d_c2b, d_c2o);

    dim3 relu2_grid_dim(
        (C2_OUT*C2_W*C2_H + THREADS_PER_BLOCK_1D - 1) / THREADS_PER_BLOCK_1D,
        1,
        BATCH_SIZE
    );
    relu_kernel<<<relu2_grid_dim, THREADS_PER_BLOCK_1D>>>(d_c2o, C2_OUT*C2_W*C2_H);

    dim3 mp2_grid_dim(
        (MP2_W + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        (MP2_H + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D,
        C2_OUT * BATCH_SIZE
    );
    maxpool_kernel<<<mp2_grid_dim, block_2d>>>(d_c2o, d_mp2, C2_OUT, C2_W, C2_H, MP2_W, MP2_H);

    // FC Layer
    // Each block processes one image, each thread processes one output neuron
    fc_kernel<<<BATCH_SIZE, FC_OUT>>>(d_mp2, d_fcw, d_fcb, d_fco);

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 5. Result Verification (for the first image in the batch) [cite: 313-315]
    std::vector<int32> final_out_batch(BATCH_SIZE * FC_OUT);
    cudaMemcpy(final_out_batch.data(), d_fco, BATCH_SIZE * FC_OUT * sizeof(int32), cudaMemcpyDeviceToHost);

    // Iterate through each image in the batch and print its prediction
    for (int batch_idx = 0; batch_idx < BATCH_SIZE; ++batch_idx) {
        int prediction = 0;
        int max_val = final_out_batch[batch_idx * FC_OUT];
        for(int i=1; i<FC_OUT; i++) {
            if(final_out_batch[batch_idx * FC_OUT + i] > max_val) {
                max_val = final_out_batch[batch_idx * FC_OUT + i];
                prediction = i;
            }
        }
        std::cout << "Predicted Digit for Image " << batch_idx << ": " << prediction << std::endl;
    }

    std::cout << "Total GPU execution time for batch prediction: " << milliseconds << " ms" << std::endl;
    std::cout << "Average GPU execution time per image: " << milliseconds / BATCH_SIZE << " ms" << std::endl;

    // Cleanup
    cudaFree(d_in); cudaFree(d_c1k); cudaFree(d_c1b);
    cudaFree(d_c2k); cudaFree(d_c2b); cudaFree(d_fcw); cudaFree(d_fcb);
    cudaFree(d_c1o); cudaFree(d_mp1); cudaFree(d_c2o); cudaFree(d_mp2); cudaFree(d_fco);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
