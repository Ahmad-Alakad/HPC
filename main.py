import numpy as np
from cv2 import cv2
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    img = cv2.imread("cat.jpg", cv2.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]

    chunks = np.array_split(img, size-1, 0)

    for s_index in range(0, size-1):
        comm.send(chunks[s_index], dest=s_index+1)

else:
    chunk = comm.recv(source=0)

    kernel = np.ones((5, 5), np.float32) / 25
    #gray = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
    dst = cv2.filter2D(chunk, -1, kernel)

    cv2.imwrite('fragment_node_{}.png'.format(rank), dst)

    comm.send(dst, dest=0)

if rank == 0:
    result = comm.recv(source=1)

    for k in range(2, size):
        result = cv2.vconcat([result, comm.recv(source=k)])

    cv2.imwrite('result_node_{}.png'.format(rank), result)
