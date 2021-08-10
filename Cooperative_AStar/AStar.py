from GlobalObjs.Graph import Node
import numpy as np
from queue import PriorityQueue

# Based off psuedo code taken from
# https://www.geeksforgeeks.org/a-search-algorithm/


class AStarNode(Node):
    def __init__(self, parent, x, y, t=-1):
        super(AStarNode, self).__init__(parent, x, y, t)
        self.h = 0  # heuristic
        self.g = 0  # distance to node from start
        self.f = np.inf  # combined distance and heuristic

    # so it can be sorted
    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f

    def __repr__(self):
        return f"Node({self.x},{self.y},{self.t},f={self.f})"

    def __str__(self):
        return self.__repr__()

    def man_dist(self, end):
        return abs(self.x - end.x) + abs(self.y - end.y)

    # def calc_f(self, end):
    #     self.f = self.g + self.man_dist(end)


# Normal AStar - I.e. ignores time dimension
class AStar:
    def __init__(self):
        pass

    @staticmethod
    def man_dist(node_1, node_2):
        return abs(node_1.x - node_2.x) + abs(node_1.y - node_2.y)

    # Assuming space region not space-time i.e. no time dimension
    @staticmethod
    def is_node_valid(grid, node):
        max_y_bound = len(grid) - 1
        max_x_bound = len(grid[0]) - 1
        x = node.x
        y = node.y

        if 0 <= x <= max_x_bound and 0 <= y <= max_y_bound and not grid[y][x]:
            return True
        else:
            return False

    @staticmethod
    def is_coord_valid(grid, x, y):
        max_y_bound = len(grid) - 1
        max_x_bound = len(grid[0]) - 1

        if 0 <= x <= max_x_bound and 0 <= y <= max_y_bound and not grid[y][x]:
            return True
        else:
            return False

    @staticmethod
    def get_path_list(end_node):
        path_list = [end_node]
        if end_node.parent:
            curr_node = end_node.parent
            path_list.append(curr_node)
            while curr_node.parent:
                curr_node = curr_node.parent
                path_list.append(curr_node)
        else:
            # Path not found
            raise Exception("end_node has no parent - no path exists.")
        path_list = list(reversed(path_list))
        return path_list

    @staticmethod
    def find_path(grid, start_pos, end_pos):
        if start_pos == end_pos:  # I.e. already at end
            print("Already at end position")
            return [AStarNode(None, start_pos[0], start_pos[1])]

        start_node = AStarNode(None, start_pos[0], start_pos[1])
        start_node.f = 0

        end_node = AStarNode(None, end_pos[0], end_pos[1])

        # start_node.calc_f(end_node) # Not required?

        # Using a queue would be faster
        open_list = []
        closed_list = []

        open_list.append(start_node)
        # open_q.put(start_node)

        while open_list:
            open_list = sorted(open_list)  # Sorted right way?

            curr_node = open_list.pop(0)
            # curr_node = open_q.get()

            # closed_list.append(curr_node)  # I think?

            x = curr_node.x
            y = curr_node.y

            children = []

            # GENERATE CHILDREN
            # get neighbours/children of current node

            # left
            if AStar.is_coord_valid(grid, x - 1, y):
                children.append(AStarNode(curr_node, x - 1, y))
            # right
            if AStar.is_coord_valid(grid, x + 1, y):
                children.append(AStarNode(curr_node, x + 1, y))
            # top
            if AStar.is_coord_valid(grid, x, y+1):
                children.append(AStarNode(curr_node, x, y+1))
            # bottom
            if AStar.is_coord_valid(grid, x, y - 1):
                children.append(AStarNode(curr_node, x, y - 1))

            curr_node.children = children  # Not sure if required since each node has a specified parent

            for child in children:
                # 1) If end node found
                if child == end_node:
                    # path found
                    end_node = child  # So it has correct parent
                    return AStar.get_path_list(end_node)

                # tmp_g = curr_node.g + 1
                child.g = curr_node.g + 1
                child.h = AStar.man_dist(child, end_node)
                child.f = child.g + child.h
                # tmp_f = (curr_node.g + 1) + AStar.man_dist(curr_node, end_node)

                # 2) If a node with the same position as child is in the open list and has a lower f
                # than child, skip this child
                tmp_list = list(filter(lambda el: el == child, open_list))
                if tmp_list and tmp_list[0].f < child.f:
                    continue

                # 3) If a node with the same position as child is in the closed list and has a lower f
                # than child, skip this child, otherwise, add the node to the open list.
                tmp_list = list(filter(lambda el: el == child, closed_list))
                if tmp_list and tmp_list[0].f < child.f:
                    continue
                else:
                    open_list.append(child)

            closed_list.append(curr_node)
        return None  # I.e. path not found


def print_grid(grid, path, start_node, end_node):
    for j in range(len(grid)):
        curr_str = "|"
        for i in range(len(grid[0])):
            curr_cell = " "
            if AStarNode(None, i, j) in path:
                curr_cell = "X"
            elif grid[j][i]:
                curr_cell = "#"
            if AStarNode(None, i, j) == start_node:
                curr_cell="S"
            elif AStarNode(None, i, j) == end_node:
                curr_cell="E"
            curr_str += curr_cell
        print(curr_str + "|")


def example():
    start_pos = [1, 1]
    end_pos = [6, 9]  # [5, 8]
    # ex_grid = [
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ]

    ex_grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    path = AStar.find_path(ex_grid, start_pos, end_pos)

    print_grid(ex_grid, path, AStarNode(None, start_pos[0], start_pos[1]),
               AStarNode(None, end_pos[0], end_pos[1]))
    # for node in path:
    #     print(f"({node.x},{node.y})")
    # print(path)


if __name__ == "__main__":
    example()
