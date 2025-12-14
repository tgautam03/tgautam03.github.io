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
<img src="https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/refs/heads/master/blog_imgs/2024-10-30-TensorCores/Figure_1.png">
<div class="thecap">Figure 1: Row major layout for storing matrices</div>
</div>

> FORTRAN stores 2D arrays in column major layout.

This means that to access an element, we need to linearize the 2D index of the element. For example, if matrix $$\bf{A}$$ is $$M \times N$$, the linearized index of element $$(6, 8)$$ can be written as $$6 \times N + 8$$.

> Generally speaking, any element $$(i, j)$$ is at the location $$i \times N + j$$ in the memory.

So far, we have discussed matrices in general. Let's now look at what precision means. 