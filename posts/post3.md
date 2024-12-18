# GPU Accelerated ML Library

I wrote a small library with PyOpenCL and NumPy to replace the obsurdly bloated popular libraries like TensorFlow and PyTorch.

So, how did I do it?

## The Idea

I wanted to create a library that was simple to use and understand, but also fast. I wanted to use OpenCL because it allows me to run my code on the GPU, which is much faster than the CPU.
If you don't know what OpenCL (if you don't know, what are you doing here then?) is, it's a framework for writing programs that execute across heterogeneous platforms consisting of CPUs, GPUs, and other processors. It's like CUDA, but it works on all platforms.

I just find installing a gigabyte worth of PyTorch crap to run a simple model overkill. Sure, GGML and GGUF are great, but if I want to train too?

## The Implementation
OpenCL is one of those things which is simple if you know C. I know a small bit of C, with enough experience to write kernels and run this website.
I used PyOpenCL to write the kernels and run them on the GPU. I don't particularly like writing in C, and Python is my 'safe language' where I can write code without worrying about memory management, segfaults, and other C things.

I used NumPy to handle the data. NumPy is a great library for handling large arrays of data, and it's fast. It's also easy to use, which is a bonus.

### Matrix Multiplication
Matrix multiplication is the most common operation in machine learning. It's also the most expensive. I wrote a kernel to multiply two matrices together. It's simple, but it's fast enough for my needs.

```cl
__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

And, it's pretty easy to read!
A basic overview of the kernel:

- The function takes two arrays, A, B, and C is the output array. M, N, and K are the dimensions of the matrices. (A is MxK, B is KxN, and C is MxN)
- For each row and column in the output matrix, it calculates the dot product of the row in A and the column in B. It stores that dot product in the output matrix.

### Activation Functions

ReLU is the most common activation function in deep learning. It's simple, and it works well. I wrote a kernel to apply ReLU to an array.

ReLU is a simple function that returns x if x > 0, and 0 otherwise. It's primarily used to introduce non-linearity into the model.

```cl
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        output[gid] = max(input[gid], 0.0f);
    }
}
```

I also wrote a kernel to calculate the derivative of the ReLU function. The derivative of ReLU is 1 if x > 0, and 0 otherwise. It's used in backpropagation to calculate the gradients.

```cl
__kernel void relu_backward(
    __global const float* input,
    __global const float* grad_output,
    __global float* grad_input,
    const int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        grad_input[gid] = input[gid] > 0 ? grad_output[gid] : 0;
    }
}
```

### Loss Functions

I also wrote a kernel to calculate the mean squared error between two arrays. MSE is a common loss function used in regression problems. (Like LLMs)

```cl
__kernel void mse(
    __global const float* pred,
    __global const float* target,
    __global float* output,
    const int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        float diff = pred[gid] - target[gid];
        output[gid] = diff * diff;
    }
}
```

All the rest is written in Python, with it just linking everything together. It's fairly boring.

## TLDR
I wrote a small library with PyOpenCL and NumPy to replace TensorFlow and PyTorch. It's fast, simple, and easy to use. It's also much smaller than the other libraries, which is a bonus.