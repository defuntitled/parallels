from mpi4py import MPI
import numpy as np

N = 1000
M = 1000
K = 1000

MASTER_RANK = 0

def generate_matrix(rows, cols):
    return np.random.randint(10, size=(rows, cols))

def main():
    comm = MPI.COMM_WORLD
    comm.Barrier()
    t_start = MPI.Wtime()
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows_per_process = N // size
    A = np.zeros((N, M), dtype=int)
    B = np.zeros((M, K), dtype=int)
    C = np.zeros((N, K), dtype=int)
    local_A = np.zeros((rows_per_process, M), dtype=int)
    local_C = np.zeros((rows_per_process, K), dtype=int)

    if rank == MASTER_RANK:
        A = generate_matrix(N, M)
        B = generate_matrix(M, K)

    
    comm.Scatter(A, local_A, root=0)
    
    comm.Bcast(B, root=0)

    for i in range(rows_per_process):
        for j in range(K):
            local_C[i, j] = np.dot(local_A[i, :], B[:, j])

    comm.Gather(local_C, C, root=0)
    
    comm.Barrier()
    t_diff = MPI.Wtime() - t_start

    if rank == MASTER_RANK:
        print(f"{N} x {M} matrix, execution time  {t_diff} sec")

if __name__ == '__main__':
    main()

# 10 x 10 matrix, execution time  0.18298342399999967 sec
# 100 x 100 matrix, execution time  0.1816392879999995 sec
# 1000 x 1000 matrix, execution time  0.9922888130000018 sec