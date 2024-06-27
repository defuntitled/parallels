from mpi4py import MPI


MESSAGE = 'kinder bueno '
MASTER_RANK = 0

def exec_master(size, rank, comm):
    print('---------------------------stating program--------------------------')
    for i in range(size):
        if i == MASTER_RANK:
            continue
        comm.send(MESSAGE, dest=i, tag=i)

def exec_worker(rank, comm):
    received_message = comm.recv(source=MASTER_RANK, tag=rank)
    print(f"Process {rank} received {received_message}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if rank == MASTER_RANK:
        exec_master(size,rank,comm)
    else:
        exec_worker(rank, comm)

if __name__ == "__main__":
    main()
