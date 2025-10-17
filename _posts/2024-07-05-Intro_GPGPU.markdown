---
layout: post
comments: false
title:  "What is GPGPU Programming?"
excerpt: "In this post, I explain the main difference between a CPU and a GPU. I also discuss why applications run faster on a GPU and how we can code a simple program that performs computations on a GPU."
date:   2024-07-05 10:00:00
---

GPGPU (General-Purpose Graphics Processing Unit) programming refers to the use of graphics processing units (GPUs) for general-purpose computing tasks like machine learning and scientific computing that are beyond graphics rendering. The next question is, how much of a difference can it make to solve a problem on a GPU compared to a CPU? To answer this, I will use simple matrix multiplication as an example, and Figure 1 shows the time comparison when the two matrices are of size $$30000 \times 30000$$.

<div class="imgcap">
<img src="../assets/2024-07-05-Intro_GPGPU/cpu_v_gpu.png" alt="this slowpoke moves"  width="800"/>
<div class="thecap">Figure 1: Runtime for a matrix multiplication on a Ryzen 7 7700 (CPU) and an RTX 3090 (GPU) </div>
</div>

> Well, that's more than $$173407\times$$ speed-up! 