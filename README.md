# CNN Inference Acceleration Across CPU, GPU, and RTL Hardware
## Cross-Architecture Performance Study

This project evaluates how a Convolutional Neural Network (CNN) performs across different computing architectures:

- **CPU** – Python baseline implementation  
- **GPU** – Custom CUDA kernel implementation  
- **RTL Hardware** – Verilog CNN accelerator  

The goal is to understand how different architectures affect **latency, throughput, and parallel execution efficiency** for neural network inference.

---

# Project Overview

Modern neural networks require large numbers of mathematical operations such as convolution and matrix multiplication. The efficiency of these operations depends heavily on the hardware architecture executing them.

This project evaluates CNN inference across three execution environments:

| Platform | Implementation |
|--------|--------|
| CPU | Python baseline |
| GPU | Custom CUDA kernels |
| RTL Hardware | Verilog CNN accelerator |

Each implementation executes the **same CNN architecture** and processes the same MNIST input data, enabling direct performance comparison.

---

# CNN Model Architecture

The CNN model performs handwritten digit classification using the **MNIST dataset**.

**Input size**
28 × 28 grayscale image


**Network structure**
Input Image (28×28)
↓
Convolution Layer 1
↓
ReLU Activation
↓
Max Pooling
↓
Convolution Layer 2
↓
ReLU Activation
↓
Max Pooling
↓
Fully Connected Layer
↓
Comparator
↓
Predicted Digit (0-9)


The comparator selects the output neuron with the maximum activation.

---

# Model Training and Data Preparation

The CNN was first trained in Python before being converted for hardware execution.

Steps:

1. Train CNN model on MNIST dataset  
2. Export trained weights and biases  
3. Convert floating-point weights to fixed-point format  
4. Generate `.mem` files used by RTL and CUDA implementations  

These `.mem` files allow the same trained model to run across all platforms.

---

# Fixed-Point Arithmetic

Hardware accelerators typically avoid floating-point operations due to complexity and power consumption.

This design uses **16-bit fixed-point representation with a 7-bit scaling factor**.

Example computation:

sum = bias << 7
sum += input × weight
output = sum >> 7


This preserves numerical accuracy while allowing efficient integer arithmetic.

---

# RTL Hardware Accelerator

The CNN accelerator was implemented in **Verilog RTL** and synthesized using FPGA tools.

Main hardware modules:

- Convolution engines  
- ReLU activation units  
- Max pooling modules  
- Fully connected computation unit  
- Comparator for final classification  

These modules form a pipeline that processes CNN inference operations sequentially.

---

# FPGA Timing Analysis

Timing analysis was performed after synthesis and implementation in Vivado.

Clock constraint:
100 MHz (10 ns clock period)


Timing summary:
Worst Negative Slack (WNS): 2.114 ns

This indicates that the design can operate faster than the constrained frequency.
Derived critical path delay:
Critical Path Delay ≈ 7.886 ns
Maximum Frequency ≈ 126.8 MHz

---

# Hardware Latency Estimation

RTL simulation shows that one inference requires approximately:
1280 clock cycles

Using the post-implementation clock period:

Latency = cycles × clock period
Latency = 1280 × 7.886 ns
Latency ≈ 10.1 µs


This represents the compute latency of the CNN accelerator.

*External memory transfer latency is not included.*

---

# CUDA GPU Implementation

The CNN was also implemented using **custom CUDA kernels**.

GPU kernels were written for:

- Convolution layers  
- ReLU activation  
- Max pooling  
- Fully connected layer  

CUDA event timers were used to measure kernel execution time.

Two configurations were evaluated:

- **Single-image inference**
- **Batch inference**

Batch processing allows GPUs to exploit massive parallelism.

---

# CPU Baseline

A Python implementation was used as the baseline reference.

CPU execution processes CNN layers sequentially.

Measured latency:
≈ 1.6 ms per image

---

# Performance Comparison

## Single Image Inference

| Platform | Latency |
|--------|--------|
| CPU | 1.627 ms |
| GPU | 0.213 ms |
| RTL Hardware | 0.010 ms |

Speedup relative to CPU:

- **RTL hardware ≈ 160× faster**
- **GPU ≈ 7.6× faster**

---

## Batch Processing

For larger batch workloads:

| Platform | Per Image Latency |
|--------|--------|
| GPU | 0.0024 ms |
| RTL Hardware | 0.010 ms |

GPU achieves higher throughput when processing large batches.

---

# Architectural Insights

Each architecture optimizes different aspects of computation.

### CPU
- Flexible programming model  
- Sequential execution  

### GPU
- Thousands of parallel threads  
- Excellent for batch workloads  

### RTL Hardware Accelerator
- Dedicated hardware pipeline  
- Deterministic low-latency inference  

Understanding these trade-offs is important when designing AI systems.

---

# Repository Structure
cnn-cross-architecture-acceleration/

├── python/ # CNN training and preprocessing

├── rtl/ # Verilog RTL implementation

├── cuda/ # CUDA GPU implementation

├── data/ # Fixed-point weights (.mem)

├── results/ # Benchmark results

├── docs/ , diagrams, presentation

├── report.ppt

└── README.md


---

# Key Contributions

This project demonstrates:

- CNN hardware acceleration using RTL design  
- Custom CUDA kernel implementation  
- Cross-architecture performance benchmarking  
- Fixed-point neural network deployment  
- Performance trade-off analysis across CPU, GPU, and hardware accelerators  

---

# Future Work

Potential improvements include:

- Shared-memory optimized CUDA kernels  
- Fully pipelined hardware accelerator  
- Larger neural network architectures  
- Deployment on physical FPGA boards  

---

