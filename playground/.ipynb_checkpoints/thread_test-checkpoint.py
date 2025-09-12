import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

def main():
    base = np.array([0,1])
    if rank == 0:
        print("Root", flush=True)
        all_arr = np.zeros(len(base)*size).astype(int)
        requests = []
        for i in range(size):
            start = i*len(base)
            end = start+len(base)
            if i == 0:
                all_arr[start:end] = np.array(base)
            else:
                req = comm.Irecv(all_arr[start:end],source=i,tag=i)
                requests.append(req)
        MPI.Request.Waitall(requests)
        process(all_arr)
    else:
        msg = base+(len(base)*rank)
        print(f"Worker: {msg}", flush=True)
        comm.Isend(msg,dest=0,tag=rank)

def process(data):
    print(f"Data: {data}",flush=True)

if __name__ == "__main__":
    main()