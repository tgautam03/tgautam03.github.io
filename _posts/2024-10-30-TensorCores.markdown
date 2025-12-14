---
layout: post
comments: false
title:  "Programming NVIDIA Tensor Cores"
excerpt: "A simple introduction explaining how to program NVIDIA's tensor cores."
date:   2024-10-30 10:00:00
---

## Introduction
Tensor cores are dedicated accelerator units (somewhat like CUDA cores) on the NVIDIA GPUs (since Volta micro-architecture) that do just one thing: Matrix Multiplication! Dedicated hardware for matrix multiplication makes sense because 90% of the computational cost from AI algorithms is dominated by matrix multiplication, such that over 90% of the computational cost comes from several matrix multiplications. When reading the official NVIDIA documentation, you'll see terms like "half-precision" and "single-precision". I'd like first to understand what this means.

## Matrices and Computer Memory
Computer memory is often presented as a linear address space through memory management techniques. This means that we cannot store a matrix in 2D form. Languages like C/C++ and Python store a 2D array of elements in a row-major layout, i.e., in the memory, 1st row is placed after the 0th row, 2nd row after 1st row, and so on.

<div class="imgcap">
<img src="/blog_imgs/2024-10-30-TensorCores/Figure_1.png">
<div class="thecap">Figure 1: Row major layout for storing matrices</div>
</div>

> FORTRAN stores 2D arrays in column major layout.

This means that to access an element, we need to linearize the 2D index of the element. For example, if matrix $$\bf{A}$$ is $$M \times N$$, the linearized index of element $$(6, 8)$$ can be written as $$6 \times N + 8$$.

> Generally speaking, any element $$(i, j)$$ is at the location $$i \times N + j$$ in the memory.

So far, we have discussed matrices in general. Let's now look at what precision means. 

## Memory Precision
The bit (binary digit) is the smallest and most fundamental digital information and computer memory unit. A byte is composed of 8 bits and is the most common unit of storage and one of the smallest addressable units of memory in most computer architectures. There are several ways to store the numbers in a matrix. The most common one is double precision (declared as `double` in C/C++). In double precision, a number is stored using 8 consecutive bytes in the memory. Another way is to store the numbers as a single precision type (declared as float in C/C++), where a number is stored using 4 consecutive bytes in the memory. This way, we can store the same number that takes up less space in memory, but we give up accuracy and the range of values we can work with.

> Single precision provides about 7 decimal digits of precision, and double precision provides about 15-17 decimal digits of precision. Single precision can represent numbers from approximately $$1.4 \times 10^{-45}$$ to $$3.4 \times 10^{38}$$, and double precision can represent numbers from approximately $$4.9 \times 10^{-324}$$ to $$1.8 \times 10^{308}$$.

<div class="imgcap">
<img src="/blog_imgs/2024-10-30-TensorCores/Figure_2.png">
<div class="thecap">Figure 2: Single vs Double Precision</div>
</div>

A step further, we have half-precision (2 Bytes) floating point numbers. These are not natively supported in standard C++. However, CUDA has an option to use half-precision (declared as `half`) and this is where tensor cores operate. 

## Tensor Cores Programming
Technically speaking, tensor cores perform matrix multiplication and accumulation, i.e., $$\bf{D} = \bf{A} \cdot \bf{B} + \bf{C}$$. However, we can initialize $$\bf{C}$$ to zeros and get matrix multiplication. When working with tensor cores, the hardware is designed specially so that the input matrices are generally half-precision (FP16) and the output is single-precision (FP32). Tensor cores are restricted to certain matrix dimensions, and Figure 3 shows an example of $$4 \times 4$$ inputs and outputs.

<div class="imgcap">
<img src="/blog_imgs/2024-10-30-TensorCores/Figure_3.png">
<div class="thecap">Figure 3: Tensor cores multiplication and accumulation (source: [NVIDIA](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/))</div>
</div>