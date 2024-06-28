#include <bits/stdc++.h>
#include <omp.h>

void task_1() // print messages
{
    int num_threads = 4;
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        std::cout << thread_id << " of " << num_threads << "\n";
    }
};

void task_2() // array summ
{
    const size_t kArraySize = 1000;

    auto start = std::chrono::steady_clock::now();

    std::srand(unsigned(std::time(nullptr)));
    auto array = std::vector<std::int32_t>{};
    array.resize(kArraySize);
    std::generate(array.begin(), array.end(), std::rand);
    std::int64_t sum = 0;
#pragma omp parallel for
    for (intptr_t i = 0; i < 1000; i++)
    {
#pragma omp atomic
        sum += array[i];
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> overall_time = end - start;
    std::cout << "exec time: " << overall_time.count() << " sec" << "\n";
    // n == 10 exec time: 0.00943996 sec
    // n == 1000 exec exec time: 0.00609419 sec
    // n == 10000000 exec time: 0.0229419 sec
};

double f(double x, double y)
{
    return std::sin(x) + std::cos(y);
}

void task_3()
{ // proizvodnaya
    const int n = 1000;
    const double h = 00.1;
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<std::vector<double>> B_dx(n, std::vector<double>(n));

    auto start = std::chrono::steady_clock::now();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = f(i * h, j * h);
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == 0 || i == n - 1)
                B_dx[i][j] = 0.0;
            else
                B_dx[i][j] = (A[i + 1][j] - A[i - 1][j]) / (2 * h);
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << " exec time: " << std::chrono::duration<double>(diff).count() << " sec" << std::endl;
    // for n == 10  exec time: 0.00507498 sec
    // for n == 100  exec time: 0.0156572 sec
    // for n == 1000  exec time: 0.0306404 sec
}

void generate_random_matrix(int rows, int cols, std::vector<std::vector<int>> &matrix)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(1, 10);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = distribution(gen);
        }
    }
}

void matrix_multiply(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C)
{
    int rows_A = A.size();
    int cols_A = A[0].size();
    int cols_B = B[0].size();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows_A; ++i)
    {
        for (int j = 0; j < cols_B; ++j)
        {
            C[i][j] = 0;
            for (int k = 0; k < cols_A; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void task_4()
{ // matrix prod

    std::vector<std::pair<int, int>> matrix_sizes = {{10, 10}, {100, 100}, {1000, 1000}};

    for (const auto &size : matrix_sizes)
    {
        int rows_A = size.first;
        int cols_A = size.second;
        int rows_B = cols_A;
        int cols_B = size.second;

        std::vector<std::vector<int>> A(rows_A, std::vector<int>(cols_A));
        std::vector<std::vector<int>> B(rows_B, std::vector<int>(cols_B));
        std::vector<std::vector<int>> C(rows_A, std::vector<int>(cols_B));

        generate_random_matrix(rows_A, cols_A, A);
        generate_random_matrix(rows_B, cols_B, B);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Умножение матриц
        matrix_multiply(A, B, C);

        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "size " << rows_A << "x" << cols_A << " time " << duration.count() << " sec." << "\n";
    }

    // size 10x10 time      0.00644378 sec.
    // size 100x100 time    0.00244767 sec.
    // size 1000x1000 time  1.48815 sec.
}

int main()
{
    task_4();
    return 0;
}