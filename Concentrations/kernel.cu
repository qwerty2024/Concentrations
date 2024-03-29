﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using std::cout;
using std::endl;

// Макросы для задания параметров теста
#define N_CONCENTRATIONS (uint32_t)5      // Количество концентраций
#define MM (uint32_t)10                   // Количество строк
#define NN (uint32_t)10                   // Количество столбцов

// Вспомогательные функции (тело ниже)
void Printer(double* data);
void Init(double *data);


int main()
{
    // основные данные хранятся в трехмерном плотном массиве
    double *data;
    data = new double[N_CONCENTRATIONS * NN * MM];

    Init(data);

    Printer(data);

    delete[] data;
    return 0;
}

void Init(double* data)
{
    for (int i = 0; i < N_CONCENTRATIONS * NN * MM; ++i) data[i] = 0.0;

    // проход по всем ячейкам
    int* n_tmp = new int[N_CONCENTRATIONS]; // временный массив для хранения номеров концентраций в ячейке
    for (int i = 0; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;

    for (uint32_t m = 0; m < MM; ++m)
    {
        for (uint32_t n = 0; n < NN; ++n)
        {
            // Рандомим количество веществ в ячейке (от 1 до N_CONCENTRATIONS)
            int count = rand() % N_CONCENTRATIONS + 1;

            // Генерация номеров концентраций в ячейке
            for (uint32_t i = 0; i < count; ++i)
            {
                int random;
                while (true)
                {
                    bool good = true;
                    random = rand() % N_CONCENTRATIONS;
                    for (uint32_t j = 0; j < count; j++)
                    {
                        if (random == n_tmp[j]) { good = false; break; }
                    }
                    if (good) break;
                }
                n_tmp[i] = random; // сохранение рандомного вещества, которое не повторяется
            }

            for (int i = count; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;

            // Запись данных в массив
            for (uint32_t i = 0; i < count; ++i)
            {
                data[n_tmp[i] * NN * MM + m * NN + n] = 1.0 / count;
            }

            // обязательно обнулить для следующей ячейки
            for (int i = 0; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;

        }
    }

    delete[] n_tmp;
}

void Printer(double* data)
{
    for (uint32_t i = 0; i < N_CONCENTRATIONS; ++i)
    {
        for (uint32_t m = 0; m < MM; ++m)
        {
            for (uint32_t n = 0; n < NN; ++n)
            {
                cout << data[i * NN * MM + m * NN + n] << " ";
            }
            cout << endl;
        }
        cout << endl << endl << endl;
    }
}






/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/