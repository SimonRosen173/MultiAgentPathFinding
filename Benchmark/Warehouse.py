import random
import os


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


if __name__ == "__main__":
    grid = txt_to_grid("maps/map_warehouse.txt", simple_layout=True)
    with open("tmp.txt", "w+") as f:
        f.write("[\n")
        for row in grid:
            f.write(str(row) + ",\n")
        f.write("]")
    print(grid)
    # x, y = get_rand_valid_point(grid)
    # print(x, y)
