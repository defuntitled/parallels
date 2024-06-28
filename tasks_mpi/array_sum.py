from mpi4py import MPI
import numpy as np

MASTER_RANK = 0
ARRAY_SIZE = 10000000
ARRAY_HIGH = '11111111'

def exec_master(comm, batch_size):
    size = comm.Get_size()
    input_data =np.array(np.random.randint(0,int(ARRAY_HIGH, 2), ARRAY_SIZE), dtype='i')
    sendbuf = input_data[:batch_size*size]
    return sendbuf, np.sum(input_data[batch_size*size:])

def exec_worker(comm,batch_size, sendbuf):
    recvbuf = np.zeros([batch_size,], dtype='i')
    comm.Scatter(sendbuf, recvbuf, root=0)
    sendbuf = np.zeros([1,], dtype='i')
    sendbuf[0] = np.sum(recvbuf)
    recvbuf = None
    if comm.Get_rank()==MASTER_RANK:
        recvbuf = np.zeros([comm.Get_size(),], dtype='i')
    comm.Gather(sendbuf, recvbuf, root=MASTER_RANK)
    return recvbuf

def main():
    comm = MPI.COMM_WORLD
    comm.Barrier()
    t_start = MPI.Wtime()
    rank = comm.Get_rank()
    size = comm.Get_size()
    batch_size = ARRAY_SIZE//size
    rsult = 0
    sendbuf = None
    if rank == MASTER_RANK:
        sendbuf, rsult = exec_master(comm,batch_size)
    recvbuf = exec_worker(comm, batch_size, sendbuf)
    comm.Barrier()
    t_diff = MPI.Wtime() - t_start
    if rank == MASTER_RANK:
        rsult+=np.sum(recvbuf)
        print(rsult, 'execution time', t_diff)
if __name__ == "__main__":
    main()

# for len == 10 answer is 1432 execution time 0.15148589799991896
# for len == 1000 answer is 127198 execution time 0.18097223899985693
# for len == 10000000 answer is 1270269037 execution time 0.29177088400001594