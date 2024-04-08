
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <set>

using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::set;

// Макросы для задания параметров теста
#define N_CONCENTRATIONS (int)5     // Количество концентраций
#define MM (int)4                   // Количество строк
#define NN (int)4                   // Количество столбцов
#define DISTRIBUTION_COEFF 0.01     // Если столько есть вещества в ячейке, то оно сдетанирует соседнюю ячекйку
#define TRANSFER_COEFF 0.99         // Сколько процетов вещества перенесется в соседнюю ячейку


#define THREAD_IN_BLOCK 256
#define LOAD_THREAD_CSR 8      // нагрузка на нить в расчетах новой длины массива для CSR


// Вспомогательные функции (тело ниже)
void Printer(double*, bool*);
void Init(double*, bool*);

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

__global__ void csr_precount(double *val, int *id, int *pos)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // крайняя левая и крайняя правая ячейки сетки, которые рассматривает данная нить (не превышет размер MM * NN)
    int left = THREAD_IN_BLOCK * idx;  
    int right = THREAD_IN_BLOCK * (idx + 1) - 1;

    // не выходим за границы сетки
    if (left >= MM * NN)
        return;

    if (right >= MM * NN)
        right = MM * NN - 1;

    __shared__ int count[THREAD_IN_BLOCK];

    // нитка идет по своим ячейкам и добавляет, если нам нужно расширить память массивов
    for (int i = 0; i <= right - left; i++)
    {
        int index_left = pos[left + i];
        int index_right = pos[left + i + 1];

        int consist[N_CONCENTRATIONS]; // Надеюсь он обнулен

        for (int j = index_left; j < index_right; j++) // заполним вещества, которые есть в ячейке
        {
            consist[id[j]] = 1;
        }

        if (left + i % MM != 0) // проверка левого соседа
        {
            for (int j = pos[left + i - 1]; j < pos[left + i]; j++)
            {
                if (consist[id[j]] != 1)
                    consist[id[j]] = -1;
            }
        }

        if (left + i % MM != MM - 1) // проверка правого соседа
        {
            for (int j = pos[left + i]; j < pos[left + i + 1]; j++)
            {
                if (consist[id[j]] != 1)
                    consist[id[j]] = -1;
            }
        }

        if (left + i >= MM) // проверка верхнего соседа
        {
            for (int j = pos[left + i - MM]; j < pos[left + i + 1 - MM]; j++)
            {
                if (consist[id[j]] != 1)
                    consist[id[j]] = -1;
            }
        }

        if (left + i < MM * NN - MM) // проверка нижнего соседа
        {
            for (int j = pos[left + i + MM]; j < pos[left + i + 1 + MM]; j++)
            {
                if (consist[id[j]] != 1)
                    consist[id[j]] = -1;
            }
        }

        // подсчитать все -1 и плюсануть счетчик нити
        for (int j = 0; j < N_CONCENTRATIONS; j++)
        {
            if (consist[j] == -1)
                count[threadIdx.x]++;
        }
    }

    __syncthreads();
    // методом сдваивания найти для этого запуска ядра количество увеличения памяти
    int res = 0;

    count[threadIdx.x];

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) 
        {
            count[threadIdx.x] += count[threadIdx.x + s];
        }

        __syncthreads();
    }

    //if (threadIdx.x == 0) output[blockIdx.x] = count[0];
}

__global__ void test_csr(double* data, double* data_new)
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

    double *data; // храним все вещества в каждой ячейке по порядку
    data = new double[size] {0.0}; 

    bool *status; // информация о том смешанная ячейка или нет
    status = new bool[NN * MM] {false}; // по умолчанию все ячейки 
    
    //double *data_new;
    //data_new = new double[size]; // каждый слой отвечает за свое вещество

    Init(data, status);

    Printer(data, status);

    //double *d_data;
    //double* d_data_new;
    //
    //cudaSetDevice(0);

    //cudaMalloc((void**)&d_data, size * sizeof(double));
    //cudaMalloc((void**)&d_data_new, size * sizeof(double));
    //cudaMemcpy(d_data, data, size * sizeof(double), cudaMemcpyHostToDevice);

    //for (int i = 0; i < 100; i++)
    //{
    //    test_default << <THREAD_IN_BLOCK, (NN * MM + THREAD_IN_BLOCK) / THREAD_IN_BLOCK >> > (d_data, d_data_new);
    //    test_default << <THREAD_IN_BLOCK, (NN * MM + THREAD_IN_BLOCK) / THREAD_IN_BLOCK >> > (d_data_new, d_data);
    //}


    //cudaMemcpy(data_new, d_data_new, size * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);

    //Printer(data_new);
    //Printer(data);

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
    //int nnz = 0;
    //double *d_val;
    //int *d_id;
    //int *d_pos; // сюда бы Long long лучше

    //for (int i = 0; i < size; i++) // самый простой тупой способ подсчета ненудевых частей
    //{
    //    if (data[i] != 0) nnz++;
    //}

    //cudaMalloc((void**)&d_val, nnz * sizeof(double));
    //cudaMalloc((void**)&d_id, nnz * sizeof(int));
    //cudaMalloc((void**)&d_pos, MM * NN * sizeof(int) + 1); // по количеству ячеек в сетке

    //convert_to_csr <<<1, 1>>> (d_data, d_val, d_id, d_pos); // чисто одно ядро делает конвертацию, конечно можно немного распараллелить, но нужно сделать доп вычисления

    //cudaDeviceSynchronize();

    //int numOutputElements; // number of elements in the output list, initialised below

    //numOutputElements = numInputElements / (THREAD_IN_BLOCK / 2);
    //if (numInputElements % (THREAD_IN_BLOCK / 2)) 
    //{
    //    numOutputElements++;
    //}

    //hostOutput = (int*)malloc(numOutputElements * sizeof(int));


    //for (int i = 0; i < numInputElements; ++i) {
    //    hostInput[i] = 1;
    //}

    //reduce0 << <gridSize, blockSize >> > (deviceInput, deviceOutput);

    //cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(int), cudaMemcpyDeviceToHost);

    //for (int ii = 1; ii < numOutputElements; ii++) {
    //    hostOutput[0] += hostOutput[ii]; //accumulates the sum in the first element
    //}

    //int sumGPU = hostOutput[0];






    //double *val = new double[nnz];
    //int *id = new int[nnz];
    //int *pos = new int[MM * NN + 1]; 
    //
    //cudaMemcpy(val, d_val, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(id, d_id, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(pos, d_pos, MM * NN * sizeof(int) + 1, cudaMemcpyDeviceToHost);




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

void Init(double* data, bool *status)
{
    // все ячейки имеют нулевое вещество
    for (int i = 0; i < MM * NN * N_CONCENTRATIONS; i += 5)
    {
        data[i] = 1.0;
    }
    
    set<pair<int, int>> tmp; // для проверки (что бы не зарандомить такуюже координату)

    for (int i = 1; i < N_CONCENTRATIONS; i++)
    {
        // рандомим координаты для вещества
        int row = rand() % MM;
        int col = rand() % NN;

        if (tmp.find(std::make_pair(row, col)) != tmp.end())
        {
            i--;
        }
        else // если заандомили неповоторяющуюся координату, то впиливаем туда вещество
        {
            tmp.insert(std::make_pair(row, col));

            status[row * NN + col] = true;

            double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX); // рандомим количество вещества, которое будет лежать в ячейке

            data[(row * NN + col) * N_CONCENTRATIONS] = 1.0 - r;
            data[(row * NN + col) * N_CONCENTRATIONS + i] = r;
        }
    }
}

void Printer(double *data, bool *status)
{
    cout << "   ";
    for (int n = 0; n < NN; ++n)
        cout << "[" << n << "] ";
    cout << endl;

    for (int m = 0; m < MM; ++m)
    {
        cout << "[" << m << "] ";
        for (int n = 0; n < NN; ++n)
        {
            cout << status[m * NN + n] << "   ";
        }
        cout << endl;
    }

    for (int m = 0; m < MM; ++m)
    {
        for (int n = 0; n < NN; ++n)
        {
            cout << "[ ";
            for (int i = 0; i < N_CONCENTRATIONS; ++i)
            {
                cout << data[(m * NN + n) * N_CONCENTRATIONS + i] << " ";
            }
            cout << "] ";
        }
        cout << endl;
    }
}

//void Init(double* data)
//{
//    for (int i = 0; i < N_CONCENTRATIONS * NN * MM; ++i) data[i] = 0.0;
//
//    // проход по всем ячейкам
//    int* n_tmp = new int[N_CONCENTRATIONS]; // временный массив для хранения номеров концентраций в ячейке
//    for (int i = 0; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;
//
//    for (int m = 0; m < MM; ++m)
//    {
//        for (int n = 0; n < NN; ++n)
//        {
//            // Рандомим количество веществ в ячейке (от 1 до N_CONCENTRATIONS)
//            int count = rand() % N_CONCENTRATIONS + 1;
//
//            // Генерация номеров концентраций в ячейке
//            for (int i = 0; i < count; ++i)
//            {
//                int random;
//                while (true)
//                {
//                    bool good = true;
//                    random = rand() % N_CONCENTRATIONS;
//                    for (int j = 0; j < count; j++)
//                    {
//                        if (random == n_tmp[j]) { good = false; break; }
//                    }
//                    if (good) break;
//                }
//                n_tmp[i] = random; // сохранение рандомного вещества, которое не повторяется
//            }
//
//            for (int i = count; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;
//
//            // Запись данных в массив
//            for (int i = 0; i < count; ++i)
//            {
//                data[n_tmp[i] * NN * MM + m * NN + n] = 1.0 / count;
//            }
//
//            // обязательно обнулить для следующей ячейки
//            for (int i = 0; i < N_CONCENTRATIONS; ++i) n_tmp[i] = -1;
//
//        }
//    }
//
//    delete[] n_tmp;
//}
//
//void Printer(double* data)
//{
//    for (int i = 0; i < N_CONCENTRATIONS; ++i)
//    {
//        for (int m = 0; m < MM; ++m)
//        {
//            for (int n = 0; n < NN; ++n)
//            {
//                cout << data[i * NN * MM + m * NN + n] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl << endl << endl;
//    }
//}






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