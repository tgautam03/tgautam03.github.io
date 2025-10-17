---
layout: post
comments: false
title:  "What is GPGPU Programming?"
excerpt: "In this post, I explain the main difference between a CPU and a GPU. I also discuss why applications run faster on a GPU and how we can code a simple program that performs computations on a GPU."
date:   2024-07-05 10:00:00
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