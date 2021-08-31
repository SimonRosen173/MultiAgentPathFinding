from Cooperative_AStar.AStar import AStarNode, AStar


class CooperativeAStar:
    def __init__(self, no_agents, grid, max_time_steps=50):
        self.no_agents = no_agents
        self.grid = grid  # I.e. grid of static obstacles
        self.resv_table = {}  # Dict/Hash table
        self.max_time_steps = max_time_steps
        pass

    def is_coord_valid(self, x, y):
        max_y_bound = len(self.grid) - 1
        max_x_bound = len(self.grid[0]) - 1

        if 0 <= x <= max_x_bound and 0 <= y <= max_y_bound and not self.grid[y][x]:
            return True
        else:
            return False

    def is_node_valid(self, x, y, t):
        if self.is_coord_valid(x,y):
            # Check resv_table
            if (x, y, t) in self.resv_table:
                return False
            else:
                return True
        else:
            return False

    # Calc path for single agent using reservation table
    # Update reservation table and return path for agent
    def calc_single_path(self, start_node: AStarNode, goal_node: AStarNode):
        # Error checking
        if not self.is_coord_valid(start_node.x, start_node.y):
            raise Exception("Start node not valid")

        if not self.is_coord_valid(goal_node.x, goal_node.y):
            raise Exception("Goal node not valid")

        if start_node.is_pos_equal(goal_node):  # I.e. already at end
            print("Already at end position")
            return [start_node]

        start_node.f = 0
        start_node.t = 0

        # Using a queue would be faster - Optimise in future
        open_list = []
        closed_list = []
        pos_to_node_map = {(start_node.x, start_node.y): start_node}

        open_list.append(start_node)
        # open_q.put(start_node)

        while open_list:
            open_list = sorted(open_list)

            curr_node = open_list.pop(0)
            # curr_node = open_q.get()

            x = curr_node.x
            y = curr_node.y
            curr_t = curr_node.t + 1

            if curr_t >= self.max_time_steps:
                print("Max time steps reached")
                break

            children = []

            # GENERATE CHILDREN
            # get neighbours/children of current node

            # left
            if self.is_node_valid(x - 1, y, curr_t):
                children.append(AStarNode(curr_node, x - 1, y, curr_t))
            # right
            if self.is_node_valid(x + 1, y, curr_t):
                children.append(AStarNode(curr_node, x + 1, y, curr_t))
            # top
            if self.is_node_valid(x, y+1, curr_t):
                children.append(AStarNode(curr_node, x, y+1, curr_t))
            # bottom
            if self.is_node_valid(x, y - 1, curr_t):
                children.append(AStarNode(curr_node, x, y - 1, curr_t))
            # wait
            if self.is_node_valid(x, y, curr_t):
                children.append(AStarNode(curr_node, x, y, curr_t))

            # curr_node.children = children  # Not sure if required since each node has a specified parent

            for child in children:
                # 1) If end node found
                if child.is_pos_equal(goal_node):
                    # path found
                    end_node = child  # So it has correct parent
                    return AStar.get_path_list(end_node)

                # tmp_g = curr_node.g + 1
                child.g = curr_node.g + 1
                child.h = AStar.man_dist(child, goal_node)
                child.f = child.g + child.h
                # tmp_f = (curr_node.g + 1) + AStar.man_dist(curr_node, end_node)

                # 2) If a node with the same position as child is in the open list and has a lower f
                # than child, skip this child

                open_node_ind = CooperativeAStar.get_node_ind(open_list, child)
                # open_node = CooperativeAStar.get_node_by_pos(open_list, child)

                if open_node_ind > -1 and open_list[open_node_ind].f < child.f:
                    continue

                # tmp_list = list(filter(lambda el: child.is_pos_equal(el), open_list))
                # tmp_list = list(filter(lambda el: el == child, open_list))
                # if tmp_list and tmp_list[0].f < child.f:
                #     continue

                # 3) If a node with the same position as child is in the closed list and has a lower f
                # than child, skip this child, otherwise, add the node to the open list.
                # Not sure if this is correct? - Check paper

                closed_node = CooperativeAStar.get_node_by_pos(closed_list, child)
                if closed_node and closed_node.f < child.f:
                    continue
                else:
                    # pos_to_node_map[(child.x, child.y)] = child
                    if open_node_ind > -1:
                        open_list[open_node_ind] = child  # Already exists in list so update
                    else:
                        open_list.append(child)

                # tmp_list = list(filter(lambda el: child.is_pos_equal(el), closed_list))
                # if tmp_list and tmp_list[0].f < child.f:
                #     continue
                # else:
                #     open_list.append(child)

            closed_list.append(curr_node)
        return []  # I.e. path not found

    def update_resv_table(self, path):
        for node in path:
            self.resv_table[(node.x, node.y, node.t)] = 1

    def calc_paths(self, start_nodes, goal_nodes):
        self.resv_table = {}
        paths = []
        for start_node, goal_node in zip(start_nodes, goal_nodes):
            curr_path = self.calc_single_path(start_node, goal_node)
            self.update_resv_table(curr_path)
            paths.append(curr_path)
        return paths

    @staticmethod
    def get_node_ind(node_list, node):
        for i, curr_node in enumerate(node_list):
            if node.is_pos_equal(curr_node):
                return i
        return -1

    @staticmethod
    def get_node_by_pos(node_list, node):
        for curr_node in node_list:
            if node.is_pos_equal(curr_node):
                return curr_node
        return None


def list_contains_coord(path, x, y):
    for node in path:
        if node.is_pos_equal(AStarNode(None, x, y)):
            return True
    return False


def print_grid(grid, path, start_node, end_node):
    for j in range(len(grid)):
        curr_str = "|"
        for i in range(len(grid[0])):
            curr_cell = " "
            if list_contains_coord(path, i, j):
                curr_cell = "X"
            elif grid[j][i]:
                curr_cell = "#"

            tmp_curr_node = AStarNode(None, i, j)
            if start_node.is_pos_equal(tmp_curr_node):
                curr_cell="S"
            elif end_node.is_pos_equal(tmp_curr_node):
                curr_cell="E"
            curr_str += curr_cell
        print(curr_str + "|")


def example():
    start_nodes = [AStarNode(None, 1, 1), AStarNode(None, 7, 3), AStarNode(None, 0, 6), AStarNode(None, 3, 6)]
    goal_nodes = [AStarNode(None, 6, 9), AStarNode(None, 2, 2), AStarNode(None, 8, 4), AStarNode(None, 1, 2)]

    # start_node = AStarNode(None, 1, 1)
    # # start_pos = [7, 2]
    # # end_pos = [5, 7]
    # goal_node = AStarNode(None, 6, 9)
    ex_grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    coop_Astar = CooperativeAStar(1, ex_grid)
    paths = coop_Astar.calc_paths(start_nodes, goal_nodes)

    # path = coop_Astar.calc_single_path(start_node, goal_node)
    # path = AStar.find_path(ex_grid, start_pos, end_pos)

    for i in range(len(paths)):
        print("#######################")
        print(f"#       AGENT {i+1}       #")
        print("#######################\n\n")

        print_grid(ex_grid, paths[i], start_nodes[i], goal_nodes[i])

    # print("#######################")
    # print("#       AGENT 2       #")
    # print("#######################\n\n")
    #
    # print_grid(ex_grid, paths[1], start_nodes[1], goal_nodes[1])
    # for node in path:
    #     print(f"({node.x},{node.y})")
    # print(path)


if __name__ == "__main__":
    example()