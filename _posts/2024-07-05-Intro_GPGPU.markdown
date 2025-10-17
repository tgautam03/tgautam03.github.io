---
layout: post
comments: false
title:  "What is GPGPU Programming?"
excerpt: "In this post, I explain the main difference between a CPU and a GPU. I also discuss why applications run faster on a GPU and how we can code a simple program that performs computations on a GPU."
date:   2024-07-05 10:00:00
hidden: true
---

GPGPU (General-Purpose Graphics Processing Unit) programming refers to the use of graphics processing units (GPUs) for general-purpose computing tasks like machine learning and scientific computing that are beyond graphics rendering. The next question is, how much of a difference can it make to solve a problem on a GPU compared to a CPU? To answer this, I will use simple matrix multiplication as an example, and Figure 1 shows the time comparison when the two matrices are of size $$30000 \times 30000$$.

<div class="imgcap">
<img src="/assets/2024-07-05-Intro_GPGPU/cpu_v_gpu.png" alt="this slowpoke moves"  width="800"/>
<div class="thecap">Figure 1: Runtime for a matrix multiplication on a Ryzen 7 7700 (CPU) and an RTX 3090 (GPU) </div>
</div>

> Well, that's more than 173407x speed-up! 

I hope you're now intrigued to learn more about how GPUs can accelerate different computational tasks. But first, I will explain the basics of GPGPU computing by answering three simple questions:

1. *What is the main difference between a CPU and a GPU?*
2. *Why do applications run faster on a GPU, and is that always true?*
3. *How can I code a simple program that performs computations on a GPU?*

## Introduction
Every year, with the release of new processors, we notice a bump in performance. This was quite significant in the early years (1980s and 1990s) when the applications relied on the advancement of processor speed, memory speed, and memory capacity. However, since 2003, this drive has slowed down because of issues like energy consumption and heat dissipation. This is evident in Figure 2, where you can see that the single thread performance (blue scatter plot) has flattened a bit in recent times (even though the number of transistors keeps increasing). 

<div class="imgcap">
<img src="/assets/2024-07-05-Intro_GPGPU/microprocessor-trend-data.png" alt="this slowpoke moves"  width="800"/>
<div class="thecap">Figure 2: History of microprocessors </div>
</div>

So, the only way to increase performance (in a significant way) now is by utilizing parallelism. Parallelism in computers refers to the ability to perform multiple computations or processes simultaneously, and there are two broad types of parallelism:
- **Task Parallelism:** Two tasks can be done independently (at the same time). A common example is data transfer, which involves transferring multiple chunks of data together.
- **Data Parallelism:** Most modern applications run slowly because of too much data. The core idea behind data parallelism is that we can work on chunks of the same data independently (at the same time). Vector addition is the simplest example for this case.

Data parallelism is the main source of scalability (and speed-up) for many modern programs, and we hear a lot about GPUs nowadays because they're designed specifically for this task. That does not mean GPUs are best at everything, and we should forget about CPUs. There are major differences in the design philosophy of CPUs and GPUs, but in the simplest terms, you can think of a

- **CPU**: As a small team of highly skilled workers where each worker can finish the task at hand quickly.
- **GPU**: As a large team of less skilled workers, each worker is a bit slower in doing their work.

Hence giving CPUs an edge over GPUs when:
- The problem is complex as the highly skilled workers will be more suitable here.
- The problem is small because highly skilled workers can finish the tasks quickly. Figure 3 demonstrates this using matrix multiplication, where I'm plotting the runtime for CPU and GPU as the matrix size increases.

<div class="imgcap">
<img src="/assets/2024-07-05-Intro_GPGPU/cpu_v_gpu_runtime.png" alt="this slowpoke moves"  width="800"/>
<div class="thecap">Figure 3: Runtime for CPU and GPU plotted as matrix size varies along the x dimension </div>
</div>

From Figure 3, it is evident that for small problems (i.e., small matrices in this case), the CPU has a shorter runtime. However, as the problem size increases, we get a crossover point where GPU starts outperforming CPU, and for really large problems, the speed-up can be as high as 173407x.

> While GPU architecture is considerably more complex, I believe it's
important to provide context before delving into the intricate details. Therefore, I will cover a basic application before discussing the ins and outs of the GPU architecture.

## GPGPU Programming Model
In this blog (and the ones in the future about GPGPU programming), I will write programs in CUDA C to exploit data parallelism. CUDA C extends C programming language with minimal new syntax and library functions to allow programs to run on GPU and CPU cores (heterogeneous computing). The structure of a CUDA C program reflects the presence of a host (CPU) and one or more devices (GPUs). It is such that the execution starts with the host, and the host assigns very specialized tasks to the device. 

> I will interchangeably use host and CPU, device and GPU.

Threads are at the heart of modern computing. A thread is a simplified view of how a processor executes a sequential program in modern computers. Adhering to the worker analogy, a thread can be seen as an individual worker, and the execution of a thread is sequential as far as a user is concerned (i.e., a worker can only do one task at a time).

When a program starts, there is a single CPU thread that sets up the stage for everything and then calls a kernel function (a function that is defined to only run on GPU). Many threads are then launched on the GPU, which executes the same kernel function and processes different parts of the data in parallel. Once the task assigned to the GPU finishes, the reigns are returned to the CPU, which can either end the program or call another kernel function for a different task.

<div class="imgcap">
<img src="/assets/2024-07-05-Intro_GPGPU/CPUnGPUexe.png" alt="this slowpoke moves"  width="800"/>
<div class="thecap"></div>
</div>

> CUDA programmers can assume that the GPU threads take very few clock cycles to generate and schedule (in stark contrast to the CPU threads, which take thousands of clock cycles to generate and schedule), owing to the hardware design choices.

The threads generated by a GPU are organized systematically. Whenever a CPU calls a kernel function, a grid is created on the GPU. This grid is divided into multiple blocks, which contain several threads (each block must have the same number of threads). The programmer decides the number of blocks in a grid and threads in a block. 

<div class="imgcap">
<img src="/assets/2024-07-05-Intro_GPGPU/GridOrganization.png" alt="this slowpoke moves"  width="800"/>
<div class="thecap"></div>
</div>

> There are several reasons for this thread organization, and I will discuss some of those in this blog post and some in the future (where I will look into more complicated examples).

## Vector Addition Example
Vector addition is the "hello world" program of parallel programming. When two vectors $$a$$ and $$b$$ of length $$n$$ are added together, the output (say vector $$c$$) contains the sum of the corresponding components.

<div class="imgcap">
<img src="/assets/2024-07-05-Intro_GPGPU/vecadd.png" alt="this slowpoke moves"  width="800"/>
<div class="thecap"></div>
</div>

### Sequential vector addition on a CPU
While running a program sequentially, there are two main hardware components that you should keep in mind:
- **CPU:** This is the workhorse that produces a thread. For a sequential program, you can assume it spawns a single thread (even if your CPU supports multi-threading).
- **Random Access Memory (RAM):** This is a storage unit where data is stored. A CPU thread accesses data from RAM and works on it.

> CPU architecture is far more complicated than this, but to keep things simple, I decided not to discuss other components like cache (as the ultimate goal is to run the code in parallel on a GPU).

The sequential code for vector addition is fairly simple. There are four steps:
1. Define the length `N` of vectors.
2. Allocate host memory (RAM) for vectors `A`, `B`, `C`, and then initialize them appropriately.
3. Perform sequential vector addition.
4. Free the host memory.

```c
// Sequential vector addition
void vec_add_cpu(float* A, float* B, float* C, int N)
{
    // Loop over elements of vectors one by one
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

int main(int argc, char const *argv[])
{
    // 1) Length of arrays
    int N = 10;
    
    // 2) Memory allocation
    float* A = (float*)malloc(N*sizeof(float));
    float* B = (float*)malloc(N*sizeof(float));
    float* C = (float*)malloc(N*sizeof(float));

    // Initialize A, B and C
    for (int i = 0; i < N; i++)
    {
        A[i] = (float)(rand() % (10 - 0 + 1)+0);
        B[i] = (float)(rand() % (10 - 0 + 1)+0);
        C[i] = 0;
    }

    // 3) Vector addition on a CPU
    vec_add_cpu(A, B, C, N);

    // 4) Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
```

> I'm assuming that you're familiar with pointers in C/C++.

Looking at the code and the description of vector addition, you can see that the operations on each element of the vectors can be performed independently. In other words, I can eliminate the loop in the function `vec_add_cpu()` (which goes over the elements one by one). Instead, I can assign each iteration to an independent GPU thread, which can all work in parallel to give the correct output.

### Parallel vector addition on a GPU
For parallel execution, we have two different hardware components:
- **GPU:** It's the workhorse that produces multiple threads that work in parallel on different subsets of data. 
- **Video Random Access Memory (VRAM):** It's a storage unit for the GPU. GPU threads can access data from this memory.

A GPU can't function independently. It's the job of a CPU to move data between RAM and VRAM and launch the kernel function (which then executes the operations in parallel on a GPU). In other words, a CPU can be seen as an instructor who manages most of the tasks and is responsible for assigning specific tasks to the GPU (where it has an advantage). 

Broadly speaking, there are eight steps to perform parallel vector addition on a GPU:
1. Define the length `N` of vectors.
2. Allocate host memory (RAM) for vectors `A`, `B`, `C`, and then initialize them appropriately.
3. Allocate device memory (VRAM) for vectors `A`, `B`, and `C`.
4. Copy data related to vectors `A` and `B` from RAM to VRAM.
5. Execute device kernel that performs parallel computations on a GPU. In this step, we define:
    - The number of blocks in the grid.
    - The number of threads in each block.
6. Copy the result of vector addition from VRAM to RAM.
7. Free device memory.
8. Free host memory.

```c
#include <stdio.h>

// CUDA error checking code
#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

// Parallel vector addition kernel
__global__ void vec_add_kernel(float* A, float* B, float* C, int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}


int main(int argc, char const *argv[])
{
    // 1) Length of arrays
    int N = 10;
    
    // 2) Memory allocation
    float* A = (float*)malloc(N*sizeof(float));
    float* B = (float*)malloc(N*sizeof(float));
    float* C = (float*)malloc(N*sizeof(float));

    // Initialize A, B and C
    for (int i = 0; i < N; i++)
    {
        A[i] = (float)(rand() % (10 - 0 + 1)+0);
        B[i] = (float)(rand() % (10 - 0 + 1)+0);
        C[i] = 0;
    }

    // 3) Allocate device memory
    float* d_A; // Device pointer for vector A
    cudaError_t err_A = cudaMalloc((void**) &d_A, N*sizeof(float)); // Device memory allocation for vector C
    CUDA_CHECK(err_A); // Checking to ensure that device memory allocation was successful
    
    float* d_B; // Device pointer for vector B
    cudaError_t err_B = cudaMalloc((void**) &d_B, N*sizeof(float)); // Device memory allocation for vector B
    CUDA_CHECK(err_B); // Checking to ensure that device memory allocation was successful
    
    float* d_C; // Device pointer for vector C
    cudaError_t err_C = cudaMalloc((void**) &d_C, N*sizeof(float)); // Device memory allocation for vector C
    CUDA_CHECK(err_C); // Checking to ensure that device memory allocation was successful

    // 4) Copy data from RAM to VRAM
    cudaError_t err_A_ = cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice); // Copying A to device memory
    CUDA_CHECK(err_A_); // Checking to ensure that RAM to VRAM copy was successful
    
    cudaError_t err_B_ = cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice); // Copying B to device memory
    CUDA_CHECK(err_B_); // Checking to ensure that RAM to VRAM copy was successful

    // 5) Kernel execution
    dim3 dim_block(4, 1, 1); // Defining the number of threads in a block
    dim3 dim_grid(ceil(N/4.0), 1, 1); // Defining the number of blocks in a grid
    vec_add_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);

    // 6) Copy back results from VRAM to RAM
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost); // Copying the result stored in C to host memory
    CUDA_CHECK(err_C_); // Checking to ensure that VRAM to RAM copy was successful

    // 7) Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 8) Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
```