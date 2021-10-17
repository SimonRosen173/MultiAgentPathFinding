import random
import os

import numpy as np
from typing import List, Tuple, Dict, Set, Optional


def print_grid(grid: list):
    for row in grid:
        row_str = " ".join([str(el) for el in row])
        print(row_str)


def txt_to_grid(file_name, simple_layout=False, use_curr_workspace=False):
    if use_curr_workspace:
        workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
        file_name = workspace_path + "/Benchmark/maps/" + file_name

    grid = None
    with open(file_name) as f:
        curr_line = f.readline()
        width = len(curr_line) - 3  # Note that '\n' is included
        # print(width)
        grid = []
        while curr_line:
            curr_line = f.readline()
            if curr_line[1] == "#":
                break
            curr_row = []
            for i in range(1, len(curr_line)-2):
                if curr_line[i] == " ":
                    curr_row.append(0)
                else:
                    if simple_layout:
                        curr_row.append(1)
                    else:
                        curr_row.append(int(curr_line[i]))
            grid.append(curr_row)
    return grid


class UniformRandomGrid:
    def __init__(self):
        abs_path = os.path.dirname(os.path.abspath(__file__))
        file_name = abs_path + "/maps/droppoff_grid.txt"
        self._static_grid = np.array(txt_to_grid(file_name))

    def get_uniform_random_grid(self, shape: Tuple[int, int], num_pickup_locs):
        # workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
        static_grid = self._static_grid
        assert num_pickup_locs < shape[0] * shape[1], "num_pickup_locs must be less than number of elements in created grid"
        rand_grid = np.zeros(shape, dtype=int)
        y_len, x_len = shape[0], shape[1]
        curr_no_locs = 0
        while curr_no_locs < num_pickup_locs:
            y = np.random.randint(0, y_len)
            x = np.random.randint(0, x_len)
            if rand_grid[y][x] == 0:
                rand_grid[y][x] = 1
                curr_no_locs += 1
        grid = np.concatenate([static_grid, rand_grid], axis=1)
        # print(np.count_nonzero(grid))
        return grid  # .tolist()


def get_uniform_random_grid(shape: Tuple[int, int], num_pickup_locs):
    # workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
    abs_path = os.path.dirname(os.path.abspath(__file__))
    file_name = abs_path + "/maps/droppoff_grid.txt"
    static_grid = np.array(txt_to_grid(file_name))
    assert num_pickup_locs < shape[0] * shape[1], "num_pickup_locs must be less than number of elements in created grid"
    rand_grid = np.zeros(shape, dtype=int)
    y_len, x_len = shape[0], shape[1]
    curr_no_locs = 0
    while curr_no_locs < num_pickup_locs:
        y = np.random.randint(0, y_len)
        x = np.random.randint(0, x_len)
        if rand_grid[y][x] == 0:
            rand_grid[y][x] = 1
            curr_no_locs += 1
    grid = np.concatenate([static_grid, rand_grid], axis=1)
    # print(np.count_nonzero(grid))
    return grid.tolist()


def get_pickup_points(grid):
    for y in range(grid):
        for x in range(grid[0]):
            pass


def get_dropoff_points(grid):
    pass


def get_rand_valid_point(grid):
    x, y = -1, -1
    valid_coord_found = False
    while not valid_coord_found:
        x = random.randint(0, len(grid[0])-1)
        y = random.randint(0, len(grid)-1)
        if grid[y][x] == 0:
            valid_coord_found = True
    return x, y


def main():
    grid = txt_to_grid("maps/map_warehouse_1.txt", simple_layout=False)
    # 5 width
    dropoff_grid = [row[:6] for row in grid]
    with open("maps/droppoff_grid.txt", "w") as f:
        f.write("#" * (2+len(dropoff_grid[0])) + "\n")
        for row in dropoff_grid:
            row_str = "".join([str(el) for el in row])
            row_str = "#" + row_str + "#\n"
            f.write(row_str)
        f.write("#" * (2+len(dropoff_grid[0])) + "\n")
    pass


if __name__ == "__main__":
    # main()
    grid = txt_to_grid("maps/droppoff_grid.txt")
    print(len(grid), len(grid[0]))
    # 560
    # 22 * 45
    urg = UniformRandomGrid()
    rand_grid = urg.get_uniform_random_grid((22, 44), 560)
    print(rand_grid)

    from GlobalObjs.GraphNX import GridGraph, plot_graph
    grid_graph = GridGraph(rand_grid, only_full_G=True)
    plot_graph(grid_graph.get_full_G(), "G.png")

    # grid = txt_to_grid("maps/map_warehouse.txt", simple_layout=True)
    # with open("tmp.txt", "w+") as f:
    #     f.write("[\n")
    #     for row in grid:
    #         f.write(str(row) + ",\n")
    #     f.write("]")
    # print(grid)
    # x, y = get_rand_valid_point(grid)
    # print(x, y)
