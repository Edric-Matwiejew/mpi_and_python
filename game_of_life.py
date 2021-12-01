from mpi4py import MPI
import numpy as np
from scipy import signal
from numpy import r_
import matplotlib.pyplot as plt

def gameStep(game_board):

    kernel = np.ones((3,3), dtype = int )
    kernel[1,1] = 0

    neighbours = signal.convolve(game_board, kernel, mode = 'same')

    game_board_old = game_board.copy()

    for i in range(game_board.shape[0]):
        for j in range(game_board.shape[1]):
            if game_board_old[i,j] == 1:
                if neighbours[i,j] < 2:
                    game_board[i,j] = 0
                elif neighbours[i,j] > 3:
                    game_board[i,j] = 0
            else:
                if neighbours[i,j] == 3:
                    game_board[i,j] = 1

    return game_board, game_board.sum()

COMM = MPI.COMM_WORLD

rank = COMM.Get_rank()
size = COMM.Get_size()

num_points = 40
rows_per_process =  num_points//size

sendbuf = []

if rank == 0:
    popluations = np.empty(size, dtype = int)
else:
    popluations = None

if rank == 0:

    A = np.random.choice(
            np.array([0, 1], dtype = int),
            size = (num_points, num_points),
            p = [0.7, 0.3])

    l=np.array([ A[i*rows_per_process:(i+1)*rows_per_process,:] for i in range(size)])
    sendbuf=l

local_game_board = np.empty((rows_per_process, num_points), dtype = int)

COMM.Scatter(
        [sendbuf, MPI.INTEGER],
        [local_game_board, MPI.INTEGER],
        root = 0
        )

num_iter = 0
max_iter = 100

total_alive = num_points*num_points

while num_iter < max_iter:

    if total_alive == 0:
        break

    if rank > 0:
        row_above = np.empty((1, num_points), dtype = int)
    
    if rank < size - 1:
        row_below = np.empty((1, num_points), dtype = int)
    
    if rank == 0:
    
        COMM.Send(
                [local_game_board[-1,:], MPI.INTEGER],
                dest = 1,
                tag = rank * 2)
    
    if rank > 0 and rank < size - 1:
    
        COMM.Recv(
                [row_above, MPI.INTEGER],
                source = rank - 1,
                tag = (rank - 1) * 2)
    
        COMM.Send(
                [local_game_board[-1,:], MPI.INTEGER],
                dest = rank + 1,
                tag = rank * 2)
    
    if rank == size - 1:
        COMM.Recv(
                [row_above, MPI.INTEGER],
                source = rank - 1,
                tag = (rank - 1)*2)
    
        COMM.Send(
                [local_game_board[0,:], MPI.INTEGER],
                dest = rank - 1,
                tag = rank * 2 + 1)
    
    if rank > 0 and rank < size - 1:
    
        COMM.Recv(
                [row_below, MPI.INTEGER],
                source = rank + 1,
                tag = (rank + 1) * 2 + 1)
    
        COMM.Send(
                [local_game_board[0,:], MPI.INTEGER],
                dest = rank - 1,
                tag = rank * 2 + 1)
    
    if rank == 0:
    
        COMM.Recv(
                [row_below, MPI.INTEGER],
                source = 1,
                tag = (rank + 1) * 2 + 1)
    
    if rank > 0 and rank < size - 1:
    
        row_below.shape=(1,num_points) 
        row_above.shape=(1,num_points)
    
        next_step, n_alive = gameStep(r_[row_above, local_game_board, row_below])
    
        local_game_board=next_step[1:-1,:]
    
    if rank == 0:
        row_below.shape=(1,num_points)
    
        next_step, n_alive = gameStep(r_[local_game_board, row_below])
    
        local_game_board=next_step[0:-1,:]
    
    if rank == size - 1:
        row_above.shape=(1,num_points)
    
        next_step, n_alive = gameStep(r_[row_above, local_game_board])
    
        local_game_board=next_step[1:,:]

    COMM.Gather(
            [n_alive, MPI.INTEGER],
            [popluations, MPI.INTEGER],
            root = 0
            )

    if rank == 0:
        total_alive=popluations.sum()
        print(f"iteration: {num_iter}, alive cells: {total_alive}", flush = True)

    total_alive=COMM.bcast(total_alive, root = 0)

    game_grid = COMM.gather(local_game_board, root = 0) 

    if rank == 0:
        game_iter = np.array(game_grid)
        game_iter = game_iter.reshape([num_points, num_points])
        plt.matshow(game_iter)
        plt.savefig(f"results/{num_iter}")
        plt.close()

    num_iter += 1


