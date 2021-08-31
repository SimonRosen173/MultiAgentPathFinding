import random


def txt_to_grid(file_name, simple_layout=False):
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
    print(grid)
    x, y = get_rand_valid_point(grid)
    print(x, y)
