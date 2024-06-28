from mpi4py import MPI
import numpy as np
import math

N = 100
DX = 0.01
MASTER_RANK = 0


def f(x, y):
    return math.sin(x) + math.cos(y)

def compute_derivative(A, local_rows, n, DX):
    B = np.zeros_like(A)
    for i in range(local_rows):
        for j in range(n):
            if 0 < j < n - 1:
                B[i, j] = (A[i, j + 1] - A[i, j - 1]) / (2 * DX)
            else:
                B[i, j] = 0.0
    return B

def main():
    comm = MPI.COMM_WORLD
    comm.Barrier()
    t_start = MPI.Wtime()
    rank = comm.Get_rank()
    size = comm.Get_size()

    batch_size = N // size

    if rank == MASTER_RANK:
        a = np.array([[f(i * DX, j * DX) for j in range(N)] for i in range(N)], dtype='d')
    else:
        a = None

    local_a = np.empty((batch_size, N), dtype='d')
    comm.Scatter(a, local_a, root=0)
        
    local_b = compute_derivative(local_a, batch_size, N, DX)
    
    if rank == MASTER_RANK:
        b = np.zeros((N, N), dtype='d')
    else:
        b = None

    comm.Gather(local_b, b, root=0)
    
    comm.Barrier()
    t_diff = MPI.Wtime() - t_start
    if rank == 0:
        print('success, execution time', t_diff)

if __name__ == "__main__":
    main()

# N x N array
# for N == 10 f(x,y) = sin(x) + cos(y) success, execution time 0.004537513999821385
# for N == 1000 f(x,y) = sin(x) + cos(y) success, execution time 0.6186991909999051
# for N == 100 f(x,y) = sin(x) + cos(y) success, execution time 0.005465150999953039