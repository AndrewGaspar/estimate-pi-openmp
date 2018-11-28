#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <mpi.h>
#include <omp.h>

using std::int64_t;

double f(double a) { return 4.0 / (1.0 + a * a); }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    auto comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    int size;
    MPI_Comm_size(comm, &size);

#pragma omp parallel
#pragma omp master
    {
        std::cout << rank << ": Num threads = " << omp_get_num_threads()
                  << std::endl;
    }

    int64_t n;
    if (rank == 0) {
        if (argc <= 1) {
            n = 100;
        } else {
            n = std::atoll(argv[1]);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT64_T, 0, comm);

    double h = 1.0 / ((double)n);

    int64_t size64 = size;
    int64_t rank64 = rank;

    auto num_local = n / size64 + (rank64 < (n % size64));

    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum) schedule(static, 2000)
    for (int64_t i = 0; i < num_local; i++) {
        sum += f(h * ((i * size64 + rank64 + 1) - 0.5));
    }

    double const partial_pi = sum * h;

    double pi = 0.0;
    MPI_Reduce(&partial_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0) {
        std::cout << std::setprecision(16) << "PI is something like: " << pi
                  << std::endl;
    }

    MPI_Finalize();
}