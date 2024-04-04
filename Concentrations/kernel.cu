
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using std::cout;
using std::endl;

// Макросы для задания параметров теста
#define N_CONCENTRATIONS (int)5     // Количество концентраций
#define MM (int)10                   // Количество строк
#define NN (int)10                  // Количество столбцов

#define THREAD_IN_BLOCK 256

// Вспомогательные функции (тело ниже)
void Printer(double* data);
void Init(double *data);

__global__ void test_default(double *data, double *data_new)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < NN * MM)
    {
        for (int i = 0; i < N_CONCENTRATIONS; i++)
        {
            int offset = i * NN * MM;

            int left = idx - 1 + offset;
            int right = idx + 1 + offset;
            int up = idx - MM + offset;
            int down = idx + MM + offset;

            int count = 1;
            double sum = data[idx + i * NN * MM];


            if (idx % MM != 0) // лево
            {
                sum += data[left];
                count++;
            }

            if (idx % MM != MM - 1) // право
            {
                sum += data[right];
                count++;
            }

            if (up - offset >= MM) // верх
            {
                sum += data[up];
                count++;
            }

            if (down - offset < MM * NN - MM) // низ
            {
                sum += data[down];
                count++;
            }

            data_new[idx + i * NN * MM] = sum / count;

            //printf("%e \n", data[idx]);
        }
    }
}

__global__ void convert_to_csr(double *data, double *val, int *id, int *pos)
{
    pos[0] = 0;
    int i = 0;
    for (int m = 0; m < MM; ++m)
    {
        for (int n = 0; n < NN; ++n)
        {
            int count = 0;
            for (int k = 0; k < N_CONCENTRATIONS; k++)
            {
                if (data[k * MM * NN + m * NN + n] != 0)
                {
                    val[i] = data[k * MM * NN + m * NN + n];
                    id[i] = k;
                    i++;
                    count++;
                }
            }

            pos[m * NN + n + 1] = pos[m * NN + n] + count;
        }
    }
}

int main()
{
    int size = N_CONCENTRATIONS * NN * MM;
    // основные данные хранятся в трехмерном плотном массиве
    double *data;
    double *data_new;
    data = new double[size]; // каждый слой отвечает за свое вещество
    data_new = new double[size]; // каждый слой отвечает за свое вещество

    Init(data);

    Printer(data);

    double *d_data;
    double* d_data_new;
    
    cudaSetDevice(0);

    cudaMalloc((void**)&d_data, size * sizeof(double));
    cudaMalloc((void**)&d_data_new, size * sizeof(double));
    cudaMemcpy(d_data, data, size * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < 100; i++)
    {
        test_default << <THREAD_IN_BLOCK, (NN * MM + THREAD_IN_BLOCK) / THREAD_IN_BLOCK >> > (d_data, d_data_new);
        test_default << <THREAD_IN_BLOCK, (NN * MM + THREAD_IN_BLOCK) / THREAD_IN_BLOCK >> > (d_data_new, d_data);
    }


    //cudaMemcpy(data_new, d_data_new, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);

    //Printer(data_new);
    Printer(data);

    // Мысль 1. 
    // Что если хранить не double, а два int, типо мантисса и порядок?
    //int a = (int)(0.12345678 * 1'000'000'000);
    //int b = (int)(1.12345678 * 1'000'000'000);
    //std::cout << a << std::endl << b << std::endl;
    // сильно падает точность... не катит


 
    // СТРУКТУРА 1
    // Есть односвязный список (можно и вектор на самом деле), каждый элемент которого хранит указатель на данные по конкретному веществу
    // Данные представлены как unordered_map<int, double>, int - номер ячейки, double - концентрация в ней
    // Теоретические плюсы: доступ, удаление, добавление в среднем за O(1)
    // Теоретические минусы: а хуй знает как поведет себя хешмапа на гпу... не понятно пока как делать копирку на гпу (мб копировать просто, а конверитровать в структуру отдельным ядром на гпу).

    
    // СТРУКТУРА 2
    // Обычные ячейки описаны простой двумерной таблицей, храним номер вещества, если смесь, то храним минус еденицу
    // В хэшмапе храним сами смешанные ячейки, где ключ - номер ячейки, а данные это структура из пар типа номер вещества и его концентрация
    // Теоретические плюсы: 
    // Теоретические минусы:
    
    
    // СТРУКТУРА 3
    // Подобие CSR хранения...
    // Первый массив - массив самих концентраций: val: 0.3, 0.7, 0.1, 0.1, 0.1, 0.7, 1.0 ...
    // Второй массив - массив id вещества:         id:   0,   3,   0,   1,   2,   3,   3 ...
    // Третий массив - массив сдвига:             pos: 0,     2,                  6,   7 ...
    // Теоретические плюсы: 
    // Теоретические минусы:
    int nnz = 0;
    double *d_val;
    int *d_id;
    int *d_pos; // сюда бы Long long лучше

    for (int i = 0; i < size; i++) // самый простой тупой способ подсчета ненудевых частей
    {
        if (data[i] != 0) nnz++;
    }

    cudaMalloc((void**)&d_val, nnz * sizeof(double));
    cudaMalloc((void**)&d_id, nnz * sizeof(int));
    cudaMalloc((void**)&d_pos, MM * NN * sizeof(int) + 1); // по количеству ячеек в сетке

    convert_to_csr <<<1, 1>>> (d_data, d_val, d_id, d_pos); // чисто одно ядро делает конвертацию, конечно можно немного распараллелить, но нужно сделать доп вычисления

    cudaDeviceSynchronize();

    double *val = new double[nnz];
    int *id = new int[nnz];
    int *pos = new int[MM * NN + 1]; 

    cudaMemcpy(val, d_val, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(id, d_id, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos, d_pos, MM * NN * sizeof(int) + 1, cudaMemcpyDeviceToHost);


    //for (int i = 0; i < nnz; i++)
    //{
    //    std::cout << "[" << id[i] << "]" << val[i] << " ";
    //}
    //std::cout << std::endl;
    //
    //for (int i = 0; i < MM * NN + 1; i++)
    //{
    //    std::cout << pos[i] << " ";
    //}
    //std::cout << std::endl;


    // СТРУКТУРА 4
    // Таблица односвязных списков.
    // Каждая ячейка - односвязный список на пары <номер вещества, его концентрация>
    // Теоретические плюсы: 
    // Теоретические минусы:
    
    
    // СТРУКТУРА 5
    // Имеется таблица ячеек unsigned long long типа, где каждый бит этого числа будет говорить о наличии каждого вещества.
    // Так же нужны еще два массива val и pos, как в структуре 3.
    // Будет небольшая экономия на памяти, но стоит ли она дополнительных вычислений при обращении?
    // Вариант похож на мертворожденным, особенно если количество веществ много...
    // Теоретические плюсы: 
    // Теоретические минусы:
    

    // СТРУКТУРА 6
    // Частично храним нули?
    // Делаем все так же как и в третьем варианте, только заранее храним определенное количество нулей, 
    // что бы если что быстро добавить в ячейку новое вещество
    // Теоретические плюсы: 
    // Теоретические минусы:



    // Придумать какие-то тесты!




    delete[] data;
    return 0;
}

void Init(double* data)
{
    for (int i = 0; i < N_CONCENTRATIONS * NN * MM; ++i) data[i] = 0.0;

    // проход по всем ячейкам
    int* n_tmp = new int[N_CONCENTRATIONS]; // временный массив для хранения номеров концентраций в ячейке
    for (int i = 0; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;

    for (int m = 0; m < MM; ++m)
    {
        for (int n = 0; n < NN; ++n)
        {
            // Рандомим количество веществ в ячейке (от 1 до N_CONCENTRATIONS)
            int count = rand() % N_CONCENTRATIONS + 1;

            // Генерация номеров концентраций в ячейке
            for (int i = 0; i < count; ++i)
            {
                int random;
                while (true)
                {
                    bool good = true;
                    random = rand() % N_CONCENTRATIONS;
                    for (int j = 0; j < count; j++)
                    {
                        if (random == n_tmp[j]) { good = false; break; }
                    }
                    if (good) break;
                }
                n_tmp[i] = random; // сохранение рандомного вещества, которое не повторяется
            }

            for (int i = count; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;

            // Запись данных в массив
            for (int i = 0; i < count; ++i)
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
    for (int i = 0; i < N_CONCENTRATIONS; ++i)
    {
        for (int m = 0; m < MM; ++m)
        {
            for (int n = 0; n < NN; ++n)
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