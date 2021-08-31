# A* Minus Dijkstra - i.e. greedy using heuristic
from GlobalObjs.Graph import Node
from MAPD import TaskAssigner
from Benchmark import Warehouse
import os
from Visualisations.Vis import Vis


class AMinusNode(Node):
    def __init__(self, parent, x, y, t=-1):
        super(Node, self).__init__(parent, x, y, t)


class AMinus:
    def __init__(self):

        pass

    @staticmethod
    def is_valid(grid: list, pos: tuple):
        x, y = pos
        return 0 <= x < len(grid[0]) and 0 <= y < len(grid) and not grid[y][x]

    @staticmethod
    def man_dist(curr_pos: tuple, goal_pos: tuple):
        return abs(goal_pos[0]-curr_pos[0]) + abs(goal_pos[1]-curr_pos[1])

    @staticmethod
    def find_path(grid: list, start_pos: tuple, goal_pos: tuple):
        count = 0  # For a stopping condition to prevent infinite loop
        max_count = len(grid)*len(grid[0])
        path = []

        curr_pos = start_pos
        while curr_pos != goal_pos:
            valid_neighbour = False
            x, y = curr_pos

            neighbours = [
                        ((x, y - 1), AMinus.man_dist((x, y-1), goal_pos)),  # Top
                        ((x, y + 1), AMinus.man_dist((x, y+1), goal_pos)),  # Bottom
                        ((x - 1, y), AMinus.man_dist((x - 1, y), goal_pos)),  # Left
                        ((x + 1, y), AMinus.man_dist((x + 1, y), goal_pos))   # Right
            ]
            neighbours = sorted(neighbours, key=lambda el: el[1])

            # Make sure I'm not trying to make code 'too smart'
            for neighbour in neighbours:
                if AMinus.is_valid(grid, neighbour[0]) and neighbour[0] not in path:
                    path.append(neighbour[0])
                    curr_pos = neighbour[0]
                    valid_neighbour = True
                    break

            # # top
            # if AMinus.is_valid(grid, (x, y - 1)):  # y-1 since (0,0) is top left
            #
            #     break
            # # bottom
            # if AMinus.is_valid(grid, (x, y + 1)):  # y-1 since (0,0) is top left
            #     break
            # # left
            # if AMinus.is_valid(grid, (x - 1, y)):  # y-1 since (0,0) is top left
            #     break
            # # right
            # if AMinus.is_valid(grid, (x + 1, y)):  # y-1 since (0,0) is top left
            #     break

            if count > max_count or not valid_neighbour:
                print("Path to goal not found")
                break
            else:
                count += 1

        return path

    @staticmethod
    def print_grid(grid, start_pos, goal_pos, path):
        for x, y in path:
            grid[y][x] = 2
        grid[start_pos[1]][start_pos[0]] = 3
        grid[goal_pos[1]][goal_pos[0]] = 4

        print("  "+"- "*(len(grid[0])+2))
        for y in range(len(grid)):
            print(f"{y:2}|", end="")
            for x in range(len(grid[0])):
                if grid[y][x] == 0:  # Empty " "
                    print("  ", end="")
                elif grid[y][x] == 1:  # Obstacle "#"
                    print("# ", end="")
                elif grid[y][x] == 2:  # Path "X"
                    print("X ", end="")
                elif grid[y][x] == 3:  # Start "S"
                    print("S ", end="")
                elif grid[y][x] == 4:  # Goal "G"
                    print("G ", end="")
            print(f"|{y:2}")
        print("  "+"- "*(len(grid[0])+2))


def warehouse_example():
    workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
    # print(workspace_path)
    grid = Warehouse.txt_to_grid(workspace_path + "/Benchmark/maps/map_warehouse.txt", simple_layout=True)
    # start_pos = Warehouse.get_rand_valid_point(grid)
    start_pos = (1, 6)
    goal_pos = (32, 15)
    # goal_pos = Warehouse.get_rand_valid_point(grid)

    print(f"start_pos: {start_pos}")
    print(f"goal_pos: {goal_pos}")
    path = AMinus.find_path(grid, start_pos, goal_pos)
    AMinus.print_grid(grid, start_pos, goal_pos, path)
    vis = Vis(grid, (1100, 500))
    vis.draw_start(start_pos)
    vis.draw_goal(goal_pos)
    vis.draw_path(path)
    vis.save_to_png("AMinusWarehouse")
    vis.window.getMouse()


def example():
    start_pos = (1, 1)
    goal_pos = (5, 3)  # [5, 8]
    # ex_grid = [[0]*10 for i in range(10)]
    # ex_grid[2][0] = 1
    # ex_grid[2][1] = 1
    # ex_grid[2][2] = 1

    ex_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    path = AMinus.find_path(ex_grid, start_pos, goal_pos)
    AMinus.print_grid(ex_grid, start_pos, goal_pos, path)

    print(f"start_pos: {start_pos}")
    print(f"goal_pos: {goal_pos}")

    vis = Vis(ex_grid, (1100, 500))
    vis.draw_start(start_pos)
    vis.draw_goal(goal_pos)
    vis.draw_path(path)
    vis.save_to_png("AMinusNonOptimal")
    vis.window.getMouse()
    # print(path)
    # ex_grid = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ]


if __name__ == "__main__":
    example()
    # warehouse_example()