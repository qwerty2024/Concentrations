
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>

using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::set;

// Опции включить тесты
//#define DEFAULT_EASY
//#define DEFAULT_HARD
#define CSR

// Дополнительные опции
//#define TEST                        // Проверяем контрльную сумму
//#define SAVE                        // Сохраняем данные, что бы потом визуализировать питон скриптом
//#define TIMER                       // Чисто вычисления, без копирования и конвертации
#define PRINTER                     // Полезно для отладки, смотрим что было до и после

// Параметры задачи
#define THREAD_IN_BLOCK 256           // нитей в блоке
#define N_CONCENTRATIONS (int)5      // Количество концентраций
#define MM (int)5                 // Количество строк
#define NN (int)5                  // Количество столбцов
#define N_STEP 1                    // Сколько раз повторить распространение волн
#define DISTRIBUTION_COEFF 0.01     // Если столько есть вещества в ячейке, то оно сдетанирует соседнюю ячекйку
#define TRANSFER_COEFF 0.9           // Сколько процетов вещества перенесется в соседнюю ячейку

// Макросы отдельные для тестов
#define LOAD_THREAD_DEFAULT_HARD 4      // количество ячеек на нить для DEFAULT_HARD
//#define LOAD_THREAD_CSR 8      // количество ячеек на нить в расчетах новой длины массива для CSR

// Функции тестов
void default_easy(const double*, const int*);
void default_hard(const double*, const int*);
void csr(const double*, const int*);

// Вспомогательные функции (тело ниже)
void Printer(const double*, const int*);
void Init(double*&, int*&);

__global__ void test_default_easy(double* data, int* status)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // номер ячейки

    if (idx < NN * MM) // так как нитей запускали больше, чем ячеек
    {
        if (status[idx] == 0) // производим расчет веществ, которые стекутся в эту ячейку от соседей
        {
            int max = 0;            // индекс самого сильного вещества
            double max_val = 0.0;   // значение самого сильного вещества

            if (idx % NN != 0) // если слева есть сосед, то смотрим его
            {
                if (status[idx - 1] == -666) // если сосед слева смешанная ячейка
                {
                    for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                    {
                        if (data[(idx - 1) * N_CONCENTRATIONS + j] > 0)
                        {
                            max = j;
                            max_val = data[(idx - 1) * N_CONCENTRATIONS + j];
                        }
                    }
                }
            }

            if (idx % NN != NN - 1) // если справа есть сосед, то смотрим его
            {
                if (status[idx + 1] == -666) // если сосед слева смешанная ячейка
                {
                    for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                    {
                        if (data[(idx + 1) * N_CONCENTRATIONS + j] > 0)
                        {
                            if (max_val < data[(idx + 1) * N_CONCENTRATIONS + j])
                            {
                                max = j;
                                max_val = data[(idx + 1) * N_CONCENTRATIONS + j];
                            }
                        }
                    }
                }
            }

            if (idx - NN >= 0) // если сверху есть сосед, то смотрим его
            {
                if (status[idx - NN] == -666) // если сверху слева смешанная ячейка
                {
                    for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                    {
                        if (data[(idx - NN) * N_CONCENTRATIONS + j] > 0)
                        {
                            if (max_val < data[(idx - NN) * N_CONCENTRATIONS + j])
                            {
                                max = j;
                                max_val = data[(idx - NN) * N_CONCENTRATIONS + j];
                            }
                        }
                    }
                }
            }

            if (idx + NN < MM * NN) // если снизу есть сосед, то смотрим его
            {
                if (status[idx + NN] == -666) // если снизу слева смешанная ячейка
                {
                    for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                    {
                        if (data[(idx + NN) * N_CONCENTRATIONS + j] > 0)
                        {
                            if (max_val < data[(idx + NN) * N_CONCENTRATIONS + j])
                            {
                                max = j;
                                max_val = data[(idx + NN) * N_CONCENTRATIONS + j];
                            }
                        }
                    }
                }
            }

            if (max != 0 && max_val > DISTRIBUTION_COEFF) // если мы нашли вещество, которое перетечет в эту ячейку и этого вещества достаточно для детонации
            {
                status[idx] = -1; // временно запишем, что эта ячейка станет детоном
                data[idx * N_CONCENTRATIONS + max] = max_val * TRANSFER_COEFF; // записали сколько вещества перетечет и умножили на коэф перетечения
            }
        }
    }
}

__global__ void test_update_default_easy(int* status)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // номер ячейки

    if (idx < NN * MM) // так как нитей запускали больше, чем ячеек
    {
        if (status[idx] == -666) status[idx] = 666;
        if (status[idx] == -1) status[idx] = -666;
    }
}

__global__ void test_default_hard(double* data, int* status)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * LOAD_THREAD_DEFAULT_HARD; // номер стартовой ячейки

    for (int k = 0; k < LOAD_THREAD_DEFAULT_HARD; k++)
    {
        if (idx < NN * MM) // так как нитей запускали больше, чем ячеек
        {
            if (status[idx] == 0) // производим расчет веществ, которые стекутся в эту ячейку от соседей
            {
                int max = 0;            // индекс самого сильного вещества
                double max_val = 0.0;   // значение самого сильного вещества

                if (idx % NN != 0) // если слева есть сосед, то смотрим его
                {
                    if (status[idx - 1] == -666) // если сосед слева смешанная ячейка
                    {
                        for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                        {
                            if (data[(idx - 1) * N_CONCENTRATIONS + j] > 0)
                            {
                                max = j;
                                max_val = data[(idx - 1) * N_CONCENTRATIONS + j];
                            }
                        }
                    }
                }

                if (idx % NN != NN - 1) // если справа есть сосед, то смотрим его
                {
                    if (status[idx + 1] == -666) // если сосед слева смешанная ячейка
                    {
                        for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                        {
                            if (data[(idx + 1) * N_CONCENTRATIONS + j] > 0)
                            {
                                if (max_val < data[(idx + 1) * N_CONCENTRATIONS + j])
                                {
                                    max = j;
                                    max_val = data[(idx + 1) * N_CONCENTRATIONS + j];
                                }
                            }
                        }
                    }
                }

                if (idx - NN >= 0) // если сверху есть сосед, то смотрим его
                {
                    if (status[idx - NN] == -666) // если сверху слева смешанная ячейка
                    {
                        for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                        {
                            if (data[(idx - NN) * N_CONCENTRATIONS + j] > 0)
                            {
                                if (max_val < data[(idx - NN) * N_CONCENTRATIONS + j])
                                {
                                    max = j;
                                    max_val = data[(idx - NN) * N_CONCENTRATIONS + j];
                                }
                            }
                        }
                    }
                }

                if (idx + NN < MM * NN) // если снизу есть сосед, то смотрим его
                {
                    if (status[idx + NN] == -666) // если снизу слева смешанная ячейка
                    {
                        for (int j = 1; j < N_CONCENTRATIONS; j++) // находим вещество, которое детонирует
                        {
                            if (data[(idx + NN) * N_CONCENTRATIONS + j] > 0)
                            {
                                if (max_val < data[(idx + NN) * N_CONCENTRATIONS + j])
                                {
                                    max = j;
                                    max_val = data[(idx + NN) * N_CONCENTRATIONS + j];
                                }
                            }
                        }
                    }
                }

                if (max != 0 && max_val > DISTRIBUTION_COEFF) // если мы нашли вещество, которое перетечет в эту ячейку и этого вещества достаточно для детонации
                {
                    status[idx] = -1; // временно запишем, что эта ячейка станет детоном
                    data[idx * N_CONCENTRATIONS + max] = max_val * TRANSFER_COEFF; // записали сколько вещества перетечет и умножили на коэф перетечения
                }
            }
        }
        else break;

        idx++;
    }
}

__global__ void test_update_default_hard(int* status)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * LOAD_THREAD_DEFAULT_HARD; // номер ячейки

    for (int k = 0; k < LOAD_THREAD_DEFAULT_HARD; k++)
    {
        if (idx < NN * MM) // так как нитей запускали больше, чем ячеек
        {
            if (status[idx] == -666) status[idx] = 666;
            if (status[idx] == -1) status[idx] = -666;
        }
        else break;

        idx++;
    }
}

//__global__ void test_default(double *data, double *data_new)
//{
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (idx < NN * MM)
//    {
//        for (int i = 0; i < N_CONCENTRATIONS; i++)
//        {
//            int offset = i * NN * MM;
//
//            int left = idx - 1 + offset;
//            int right = idx + 1 + offset;
//            int up = idx - MM + offset;
//            int down = idx + MM + offset;
//
//            int count = 1;
//            double sum = data[idx + i * NN * MM];
//
//
//            if (idx % MM != 0) // лево
//            {
//                sum += data[left];
//                count++;
//            }
//
//            if (idx % MM != MM - 1) // право
//            {
//                sum += data[right];
//                count++;
//            }
//
//            if (up - offset >= MM) // верх
//            {
//                sum += data[up];
//                count++;
//            }
//
//            if (down - offset < MM * NN - MM) // низ
//            {
//                sum += data[down];
//                count++;
//            }
//
//            data_new[idx + i * NN * MM] = sum / count;
//
//            //printf("%e \n", data[idx]);
//        }
//    }
//}
//__global__ void csr_precount(double *val, int *id, int *pos)
//{
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    
//    // крайняя левая и крайняя правая ячейки сетки, которые рассматривает данная нить (не превышет размер MM * NN)
//    int left = THREAD_IN_BLOCK * idx;  
//    int right = THREAD_IN_BLOCK * (idx + 1) - 1;
//
//    // не выходим за границы сетки
//    if (left >= MM * NN)
//        return;
//
//    if (right >= MM * NN)
//        right = MM * NN - 1;
//
//    __shared__ int count[THREAD_IN_BLOCK];
//
//    // нитка идет по своим ячейкам и добавляет, если нам нужно расширить память массивов
//    for (int i = 0; i <= right - left; i++)
//    {
//        int index_left = pos[left + i];
//        int index_right = pos[left + i + 1];
//
//        int consist[N_CONCENTRATIONS]; // Надеюсь он обнулен
//
//        for (int j = index_left; j < index_right; j++) // заполним вещества, которые есть в ячейке
//        {
//            consist[id[j]] = 1;
//        }
//
//        if (left + i % MM != 0) // проверка левого соседа
//        {
//            for (int j = pos[left + i - 1]; j < pos[left + i]; j++)
//            {
//                if (consist[id[j]] != 1)
//                    consist[id[j]] = -1;
//            }
//        }
//
//        if (left + i % MM != MM - 1) // проверка правого соседа
//        {
//            for (int j = pos[left + i]; j < pos[left + i + 1]; j++)
//            {
//                if (consist[id[j]] != 1)
//                    consist[id[j]] = -1;
//            }
//        }
//
//        if (left + i >= MM) // проверка верхнего соседа
//        {
//            for (int j = pos[left + i - MM]; j < pos[left + i + 1 - MM]; j++)
//            {
//                if (consist[id[j]] != 1)
//                    consist[id[j]] = -1;
//            }
//        }
//
//        if (left + i < MM * NN - MM) // проверка нижнего соседа
//        {
//            for (int j = pos[left + i + MM]; j < pos[left + i + 1 + MM]; j++)
//            {
//                if (consist[id[j]] != 1)
//                    consist[id[j]] = -1;
//            }
//        }
//
//        // подсчитать все -1 и плюсануть счетчик нити
//        for (int j = 0; j < N_CONCENTRATIONS; j++)
//        {
//            if (consist[j] == -1)
//                count[threadIdx.x]++;
//        }
//    }
//
//    __syncthreads();
//    // методом сдваивания найти для этого запуска ядра количество увеличения памяти
//    //int res = 0;
//
//    //count[threadIdx.x];
//
//    for (int s = 1; s < blockDim.x; s *= 2) 
//    {
//        if (threadIdx.x % (2 * s) == 0) 
//        {
//            count[threadIdx.x] += count[threadIdx.x + s];
//        }
//
//        __syncthreads();
//    }
//
//    //if (threadIdx.x == 0) output[blockIdx.x] = count[0];
//}
//
//__global__ void test_csr(double* data, double* data_new)
//{
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (idx < NN * MM)
//    {
//        for (int i = 0; i < N_CONCENTRATIONS; i++)
//        {
//            int offset = i * NN * MM;
//
//            int left = idx - 1 + offset;
//            int right = idx + 1 + offset;
//            int up = idx - MM + offset;
//            int down = idx + MM + offset;
//
//            int count = 1;
//            double sum = data[idx + i * NN * MM];
//
//
//            if (idx % MM != 0) // лево
//            {
//                sum += data[left];
//                count++;
//            }
//
//            if (idx % MM != MM - 1) // право
//            {
//                sum += data[right];
//                count++;
//            }
//
//            if (up - offset >= MM) // верх
//            {
//                sum += data[up];
//                count++;
//            }
//
//            if (down - offset < MM * NN - MM) // низ
//            {
//                sum += data[down];
//                count++;
//            }
//
//            data_new[idx + i * NN * MM] = sum / count;
//
//            //printf("%e \n", data[idx]);
//        }
//    }
//}
//
//__global__ void convert_to_csr(double *data, double *val, int *id, int *pos)
//{
//    pos[0] = 0;
//    int i = 0;
//    for (int m = 0; m < MM; ++m)
//    {
//        for (int n = 0; n < NN; ++n)
//        {
//            int count = 0;
//            for (int k = 0; k < N_CONCENTRATIONS; k++)
//            {
//                if (data[k * MM * NN + m * NN + n] != 0)
//                {
//                    val[i] = data[k * MM * NN + m * NN + n];
//                    id[i] = k;
//                    i++;
//                    count++;
//                }
//            }
//
//            pos[m * NN + n + 1] = pos[m * NN + n] + count;
//        }
//    }
//}

int main()
{
    if ((long long)MM * (long long)NN * (long long)N_CONCENTRATIONS > 2'000'000'000)
    {
        cout << "The cell limit has been exceeded." << endl;
        return 0;
    }

    cudaSetDevice(0);

    // исходные данные (что бы у всех тестов были одинаковые)
    double *data = nullptr;   // храним все вещества в каждой ячейке по порядку
    int *status = nullptr;    // информация о том смешанная ячейка или нет (-666 означает смешанная)

    Init(data, status); // заполнение


#ifdef DEFAULT_EASY
    default_easy(data, status);
#endif

#ifdef DEFAULT_HARD
    default_hard(data, status);
#endif

#ifdef CSR
    csr(data, status);
#endif

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
    //for (int i = 0; i < size; i++) // самый простой тупой способ подсчета ненулевых частей
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


    // чисточка
    delete[] data;
    delete[] status;

    return 0;
}

void Init(double *&data, int *&status)
{  
    int size = N_CONCENTRATIONS * NN * MM;
    
    cout << "Memory consumption for init: " << ((double)((long long)size * sizeof(data[0]) + (long long)NN * MM * sizeof(status[0]))) / (1024 * 1024 * 1024) << " Gbyte" << endl;

    data = new double[size] {0.0};
    status = new int[NN * MM] {0}; // по умолчанию все ячейки 

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

            status[row * NN + col] = -666; // объявили ячейку смешанной

            double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX); // рандомим количество вещества, которое будет лежать в ячейке

            data[(row * NN + col) * N_CONCENTRATIONS + i] = r;
        }
    }
}

void Printer(const double *data, const int *status)
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

void default_easy(const double* _data, const int* _status)
{
#ifdef TIMER
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    int size = N_CONCENTRATIONS * NN * MM;

    double *data; // храним все вещества в каждой ячейке по порядку
    data = new double[size];

    int *status; // информация о том смешанная ячейка или нет
    status = new int[NN * MM]; // по умолчанию все ячейки 

    // копирование из исходных данных
    for (int i = 0; i < size; i++)
    {
        data[i] = _data[i];
    }

    for (int i = 0; i < NN * MM; i++)
    {
        status[i] = _status[i];
    }

#ifdef PRINTER
    Printer(data, status);
#endif

    double *d_data;
    int *d_status;

    cudaMalloc((void**)&d_data, size * sizeof(double));
    cudaMalloc((void**)&d_status, MM * NN * sizeof(int));

    cudaMemcpy(d_data, data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_status, status, MM * NN * sizeof(int), cudaMemcpyHostToDevice);

#ifdef TIMER
    cudaEventRecord(start, 0);
#endif

    for (int i = 0; i < N_STEP; i++)
    {
        test_default_easy << < (NN * MM + THREAD_IN_BLOCK) / THREAD_IN_BLOCK, THREAD_IN_BLOCK >> > (d_data, d_status);
        test_update_default_easy << < (NN * MM + THREAD_IN_BLOCK) / THREAD_IN_BLOCK, THREAD_IN_BLOCK >> > (d_status);

#ifdef SAVE
        std::string name = "data/DEFAULT_EASY/" + std::to_string(i + 1) + ".txt";
        std::ofstream out;          // поток для записи
        out.open(name);      // открываем файл для записи
        if (out.is_open())
        {
            //out << "Hello World!" << std::endl;
            cudaMemcpy(data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);

            for (int k = 0; k < MM * NN; k++)
            {
                int item = 0;
                double val = 0.0;

                for (int f = 1; f < N_CONCENTRATIONS; f++)
                {
                    if (data[k * N_CONCENTRATIONS + f] > 0.00000000001)
                    {
                        item = f;
                        val = data[k * N_CONCENTRATIONS + f];
                        break;
                    }
                }
                out << item << " " << val << std::endl;
            }
        }
        out.close();
#endif

    }

#ifdef TIMER
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); 

    cout << "DEFAULT EASY TIME = " << elapsedTime / 1000 << endl;
#endif

#ifdef TEST

    cudaMemcpy(data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(status, d_status, MM * NN * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef PRINTER
    Printer(data, status);
#endif

    double res = 0;
    for (int i = 0; i < size; i++)
        res += data[i];
    cout << res << endl;
#endif

    delete[] data;
    delete[] status;
    cudaFree(d_data);
    cudaFree(d_status);

#ifdef TIMER
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}

void default_hard(const double* _data, const int* _status)
{
#ifdef TIMER
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    int size = N_CONCENTRATIONS * NN * MM;

    double* data; // храним все вещества в каждой ячейке по порядку
    data = new double[size];

    int* status; // информация о том смешанная ячейка или нет
    status = new int[NN * MM]; // по умолчанию все ячейки 

    // копирование из исходных данных
    for (int i = 0; i < size; i++)
    {
        data[i] = _data[i];
    }

    for (int i = 0; i < NN * MM; i++)
    {
        status[i] = _status[i];
    }

#ifdef PRINTER
    Printer(data, status);
#endif

    double* d_data;
    int* d_status;

    cudaMalloc((void**)&d_data, size * sizeof(double));
    cudaMalloc((void**)&d_status, MM * NN * sizeof(int));

    cudaMemcpy(d_data, data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_status, status, MM * NN * sizeof(int), cudaMemcpyHostToDevice);

#ifdef TIMER
    cudaEventRecord(start, 0);
#endif

    for (int i = 0; i < N_STEP; i++)
    {
        test_default_hard << < ((NN * MM + LOAD_THREAD_DEFAULT_HARD) / LOAD_THREAD_DEFAULT_HARD + THREAD_IN_BLOCK) / THREAD_IN_BLOCK, THREAD_IN_BLOCK >> > (d_data, d_status);
        test_update_default_hard << < ((NN * MM + LOAD_THREAD_DEFAULT_HARD) / LOAD_THREAD_DEFAULT_HARD + THREAD_IN_BLOCK) / THREAD_IN_BLOCK, THREAD_IN_BLOCK >> > (d_status);
    }

#ifdef TIMER
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); 

    cout << "DEFAULT HARD TIME = " << elapsedTime / 1000 << endl;
#endif

#ifdef TEST

    cudaMemcpy(data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(status, d_status, MM * NN * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef PRINTER
    Printer(data, status);
#endif

    double res = 0;
    for (int i = 0; i < size; i++)
        res += data[i];
    cout << res << endl;

#endif

    delete[] data;
    delete[] status;
    cudaFree(d_data);
    cudaFree(d_status);

#ifdef TIMER
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}

void csr(const double* _data, const int* _status)
{
#ifdef TIMER
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

#ifdef PRINTER
    Printer(_data, _status);
#endif

    int size = N_CONCENTRATIONS * NN * MM;

    int nnz = 0;


    double* val;
    int* id;
    int* pos;

    double* d_val;
    int* d_id;
    int* d_pos;


    for (int i = 0; i < size; i++) // самый простой тупой способ подсчета ненулевых частей
    {
        if (_data[i] != 0) nnz++; 
    }

    val = new double[nnz];
    id = new int[nnz];
    pos = new int[MM * NN + 1];


    pos[0] = 0;
    int ind = 0;
    for (int m = 0; m < MM; ++m)
    {
        for (int n = 0; n < NN; ++n)
        {
            int count = 0;
            for (int k = 0; k < N_CONCENTRATIONS; k++)
            {
                
                if (_data[m * NN * N_CONCENTRATIONS + n * N_CONCENTRATIONS + k] != 0)
                {
                    val[ind] = _data[m * NN * N_CONCENTRATIONS + n * N_CONCENTRATIONS + k];
                    id[ind] = n;
                    ind++;
                    count++;
                }
            }

            pos[m * NN + n + 1] = pos[m * NN + n] + count;
        }
    }

    cudaMalloc((void**)&d_val, nnz * sizeof(double));
    cudaMalloc((void**)&d_id, nnz * sizeof(int));
    cudaMalloc((void**)&d_pos, MM * NN * sizeof(int) + 1); // по количеству ячеек в сетке

    cudaMemcpy(d_val, val, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_id, id, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos, MM * NN * sizeof(double) + 1, cudaMemcpyHostToDevice);

    //for (int i = 0; i < nnz; i++)
    //    cout << val[i] << " ";
    //cout << endl;
    //
    //
    //for (int i = 0; i < nnz; i++)
    //    cout << id[i] << " ";
    //cout << endl;
    //
    //
    //for (int i = 0; i < MM * NN + 1; i++)
    //    cout << pos[i] << " ";
    //cout << endl;

    delete[] val;
    delete[] id;
    delete[] pos;

    //convert_to_csr <<<1, 1>>> (d_data, d_val, d_id, d_pos); // чисто одно ядро делает конвертацию, конечно можно немного распараллелить, но нужно сделать доп вычисления
    //cudaDeviceSynchronize();





    // TODO HERE













    cudaFree(d_val);
    cudaFree(d_id);
    cudaFree(d_pos);


//    double* data; // храним все вещества в каждой ячейке по порядку
//    data = new double[size];
//
//    int* status; // информация о том смешанная ячейка или нет
//    status = new int[NN * MM]; // по умолчанию все ячейки 
//
//    // копирование из исходных данных
//    for (int i = 0; i < size; i++)
//    {
//        data[i] = _data[i];
//    }
//
//    for (int i = 0; i < NN * MM; i++)
//    {
//        status[i] = _status[i];
//    }
//
//#ifdef PRINTER
//    Printer(data, status);
//#endif
//
//    double* d_data;
//    int* d_status;
//
//    cudaMalloc((void**)&d_data, size * sizeof(double));
//    cudaMalloc((void**)&d_status, MM * NN * sizeof(int));
//
//    cudaMemcpy(d_data, data, size * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_status, status, MM * NN * sizeof(int), cudaMemcpyHostToDevice);
//
//#ifdef TIMER
//    cudaEventRecord(start, 0);
//#endif
//
//    for (int i = 0; i < N_STEP; i++)
//    {
//        test_default_hard << < ((NN * MM + LOAD_THREAD_DEFAULT_HARD) / LOAD_THREAD_DEFAULT_HARD + THREAD_IN_BLOCK) / THREAD_IN_BLOCK, THREAD_IN_BLOCK >> > (d_data, d_status);
//        test_update_default_hard << < ((NN * MM + LOAD_THREAD_DEFAULT_HARD) / LOAD_THREAD_DEFAULT_HARD + THREAD_IN_BLOCK) / THREAD_IN_BLOCK, THREAD_IN_BLOCK >> > (d_status);
//    }
//
//#ifdef TIMER
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    float elapsedTime;
//    cudaEventElapsedTime(&elapsedTime, start, stop);
//
//    cout << "DEFAULT HARD TIME = " << elapsedTime / 1000 << endl;
//#endif
//
//#ifdef TEST
//
//    cudaMemcpy(data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);
//    cudaMemcpy(status, d_status, MM * NN * sizeof(int), cudaMemcpyDeviceToHost);
//
//#ifdef PRINTER
//    Printer(data, status);
//#endif
//
//    double res = 0;
//    for (int i = 0; i < size; i++)
//        res += data[i];
//    cout << res << endl;
//
//#endif
//
//    delete[] data;
//    delete[] status;
//    cudaFree(d_data);
//    cudaFree(d_status);
//
//#ifdef TIMER
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//#endif
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