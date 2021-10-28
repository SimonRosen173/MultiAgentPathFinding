# distutils: language=c++

from libcpp.queue cimport queue

import numpy as np
cimport numpy as np

# import queue

DTYPE = int  # np.int
ctypedef np.int_t DTYPE_t


def get_no_reachable_locs(np.ndarray[DTYPE_t, ndim=2] grid, int start_x, int start_y):
    cdef queue[int] q_x
    cdef queue[int] q_y

    cdef int y_len = grid.shape[0]
    cdef int x_len = grid.shape[1]
    cdef int x
    cdef int y
    cdef int curr_x
    cdef int curr_y

    cdef np.ndarray seen = np.zeros([y_len, x_len], dtype=DTYPE)

    seen[start_y, start_x] = 1
    q_x.push(start_x)
    q_y.push(start_y)

    while not q_x.empty():
        x = q_x.front()
        y = q_y.front()
        q_x.pop()
        q_y.pop()
        # Get neighbours
        # Left
        curr_y = y
        if x - 1 > 0:
            curr_x = x - 1
            # Curr neighbour has not been seen yet
            if seen[curr_y, curr_x] == 0:
                seen[curr_y, curr_x] = 1

                # If neighbour is unobstructed
                if grid[curr_y, curr_x] == 0:
                    q_x.push(curr_x)
                    q_y.push(curr_y)

        # Right
        if x + 1 < x_len:
            curr_x = x + 1
            # Curr neighbour has not been seen yet
            if seen[curr_y, curr_x] == 0:
                seen[curr_y, curr_x] = 1

                # If neighbour is unobstructed
                if grid[curr_y, curr_x] == 0:
                    q_x.push(curr_x)
                    q_y.push(curr_y)

        # Up
        curr_x = x
        if y - 1 > 0:
            curr_y = y - 1
            # Curr neighbour has not been seen yet
            if seen[curr_y, curr_x] == 0:
                seen[curr_y, curr_x] = 1

                # If neighbour is unobstructed
                if grid[curr_y, curr_x] == 0:
                    q_x.push(curr_x)
                    q_y.push(curr_y)

        # Down
        if y + 1 < y_len:
            curr_y = y + 1
            # Curr neighbour has not been seen yet
            if seen[curr_y, curr_x] == 0:
                seen[curr_y, curr_x] = 1

                # If neighbour is unobstructed
                if grid[curr_y, curr_x] == 0:
                    q_x.push(curr_x)
                    q_y.push(curr_y)

        # # Left
        # if x - 1 > 0:
        #     if seen[y, x-1] == 0:
        #         seen[y, x-1] = 1
        #
        #         # if grid[y, x-1]:
        #         q_x.push(x-1)
        #         q_y.push(y)
        #
        # # Right
        # if x + 1 < x_len:
        #     if seen[y, x+1] == 0:
        #         q_x.push(x+1)
        #         q_y.push(y)
        #         seen[y, x+1] = 1
        #
        # # Up
        # if y - 1 > 0:
        #     if seen[y-1, x] == 0:
        #         q_x.push(x)
        #         q_y.push(y-1)
        #         seen[y-1, x] = 1
        #
        # # Down
        # if y + 1 < y_len:
        #     if seen[y+1, x] == 0:
        #         q_x.push(x)
        #         q_y.push(y+1)
        #         seen[y+1, x] = 1

    cdef int no_reachable_locs

    no_reachable_locs = np.count_nonzero(seen)
    return no_reachable_locs

    # cdef queue[int] q
    # q.push(1)
    # q.push(2)
    # print(q.front())
    # q.push(1)
    # cdef int x = 2
    # # print(q)
    # print(q.pop())
