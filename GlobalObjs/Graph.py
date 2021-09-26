# # Node for space-time graph
# class Node:
#     def __init__(self, parent, x, y, t):
#         self.x = x
#         self.y = y
#         self.t = t
#         self.parent = parent
#         self.edges = []  # list of tuples of edge length and node
#         self.children = []
#
#     def add_node(self, edge_len, node_ind):
#         self.edges.append((edge_len, node_ind))
#
#     def is_pos_equal(self, other):
#         return self.x == other.x and self.y == other.y
#
#     def __eq__(self, other):
#         if self.x == other.x and self.y == other.y and self.t == other.t:
#             return True
#         else:
#             return False
#
#     def __repr__(self):
#         return self.__str__()
#
#     def __str__(self):
#         return f"Node({self.x},{self.y},{self.t})"
#         # return f"NODE ({self.x},{self.y},{self.t}) : parent = {self.parent}, edges = {str(self.edges)}"
import pydot
import os
from Benchmark import Warehouse


class Node:
    def __init__(self, parent, x, y):
        self.x = x
        self.y = y
        self.parent = parent
        self.edges = []

    def clear_edges(self):
        self.edges = []

    def add_edge(self, neighbour, weight: int):
        self.edges.append((neighbour, weight))

    def clone(self):
        tmp_node = Node(self.parent, self.x, self.y)
        tmp_node.edges = self.edges
        return tmp_node

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Node({self.x},{self.y})"


class SpaceTimeNode(Node):
    def __init__(self, parent, x, y, t):
        super().__init__(parent, x, y)
        self.t = t

    def is_pos_equal(self, other):
        return self.x == other.x and self.y == other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.t == other.t


class Graph:
    @staticmethod
    def man_dist_tup(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def man_dist_node(node1, node2):
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)

    @staticmethod
    def create_from_dict_coords(graph_dict):
        nodes = {key: Node(None, key[0], key[1]) for key in graph_dict.keys()}
        # structure
        for key in graph_dict:
            edges_pos = graph_dict[key]
            curr_node = nodes[key]
            for edge_pos in edges_pos:
                edge_node = nodes[edge_pos]
                weight = Graph.man_dist_node(curr_node, edge_node)
                curr_node.edges.append((edge_node, weight))

        return nodes

    @staticmethod
    def create_from_grid_dense(grid):
        x_len = len(grid[0])
        y_len = len(grid)
        nodes = {}
        last_node = None

        for y in range(y_len):
            for x in range(x_len - 1):
                curr_node = None
                next_node = None

                if (x, y) in nodes:
                    curr_node = nodes[(x, y)]
                elif not grid[y][x]:
                    curr_node = Node(None, x, y)
                    nodes[(x, y)] = curr_node

                if not grid[y][x+1]:
                    next_node = Node(None, x+1, y)
                    nodes[(x+1, y)] = next_node
                    if curr_node is not None:
                        curr_node.add_edge(next_node, 1)
                        next_node.add_edge(curr_node, 1)

        for x in range(x_len):
            for y in range(y_len - 1):
                curr_node = None
                next_node = None

                if (x, y) in nodes:
                    curr_node = nodes[(x, y)]

                if (x, y+1) in nodes:
                    curr_node = nodes[(x, y+1)]

                if curr_node is not None and next_node is not None:
                    curr_node.add_edge(next_node, 1)
                    next_node.add_edge(curr_node, 1)

        return nodes

    @staticmethod
    def create_from_grid_sparse(grid):
        x_len = len(grid[0])
        y_len = len(grid)
        nodes = {}
        last_node = None

        # scan horizontally
        for y in range(y_len):
            last_node = None
            for x in range(x_len):
                if last_node is None and not grid[y][x]:
                    last_node = Node(None, x, y)
                    nodes[(x,y)] = last_node
                    if x + 1 >= x_len or (x + 1 < x_len and grid[y][x + 1]):
                        nodes[(x, y)] = last_node
                else:
                    if x + 1 >= x_len or (x+1 < x_len and grid[y][x+1]):
                        curr_node = Node(None, x, y)
                        weight = Graph.man_dist_node(last_node, curr_node)
                        last_node.edges.append((curr_node, weight))
                        curr_node.edges.append((last_node, weight))
                        last_node = curr_node
                        nodes[(x, y)] = last_node

        # scan vertically
        for x in range(x_len):
            for y in range(y_len-1):
                if not grid[y][x] and not grid[y+1][x]:
                    curr_node = nodes[(x, y)]
                    next_node = nodes[(x, y+1)]
                    curr_node.edges.append((next_node, 1))
                    next_node.edges.append((curr_node, 1))
        return nodes

    @staticmethod
    def nodes_to_dot_file(nodes_dict, file_name, rank_by_y=False):
        with open(file_name, "w+") as f:
            f.write("digraph {\n")
            for key in nodes_dict:
                f.write(f"{key[0]}_{key[1]} [label=\"({key[0]},{key[1]})\"]\n")
            for key in nodes_dict:
                curr_node = nodes_dict[key]
                edges = curr_node.edges
                for edge_node, weight in edges:
                    f.write(f"{curr_node.x}_{curr_node.y} -> {edge_node.x}_{edge_node.y} [label=\"{weight}\", weight=\"{weight}\"];\n")
            f.write("}")
        pass


def dot_file_to_png(dot_file_name, png_file_name=None):
    graph, = pydot.graph_from_dot_file(dot_file_name)
    if png_file_name is None:
        png_file_name = ".".join(dot_file_name.split(".")[:-1]) + ".png"

    graph.write_png(png_file_name)


# class Graph:
#     def __init__(self, head: Node, x_size: int, y_size: int):
#         self.head: Node = head
#         self.x_size = x_size
#         self.y_size = y_size
#         self.nodes = []
#         self.edges = []  # Not sure if required
#
#     def init_from_edge_dict(self, node_dict):
#         self.nodes = [None]*len(node_dict)  # No idea if this leads to speed ups
#         for i, el_ind in enumerate(node_dict):
#             el = node_dict[el_ind]
#             tmp_node = Node(None, el[0][0], el[0][1], el[0][2])
#             self.nodes[i] = tmp_node
#
#         for i, el_ind in enumerate(node_dict):
#             el = node_dict[el_ind]
#             curr_node: Node = self.nodes[i]  # Typing for IDE
#             neighbour_inds = el[1]
#             for neighbour_ind in neighbour_inds:
#                 curr_node.edges.append((1, neighbour_ind))
#
#             if i == 0:
#                 self.head = curr_node
#
#     ##################
#     # Visualisations #
#     ##################
#     def graph_to_dot_file(self, file_name):
#         pass
#
#     # Grid for each time step -> How useful?
#     def graph_to_grids(self):
#         pass

def test_graph():
    # ex_dict = {
    #     (0, 0): [(0, 0), (0, 10)],
    #     (0, 10): [(0, 0), (0, 20)],
    #     (10, 0): [],
    # }
    workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
    # print(workspace_path)
    grid = Warehouse.txt_to_grid(workspace_path + "/Benchmark/maps/map_warehouse.txt", simple_layout=True)

    Warehouse.print_grid(grid)

    nodes = Graph.create_from_grid_dense(grid)
    #
    # nodes = Graph.create_from_dict_coords(ex_dict)
    Graph.nodes_to_dot_file(nodes, "graph.dot")
    dot_file_to_png("graph.dot")


if __name__ == "__main__":
    test_graph()
    # tmp_dict = {
    #     0: ([0, 0, 0], [1, 2, 3]),
    #     1: ([1, 2, 0], [2]),
    #     2: ([3, 4, 0], [1, 2]),
    #     3: ([5, 6, 0], [0, 2])
    # }
    # tmp_graph = Graph(None, 10, 10)
    # tmp_graph.init_from_edge_dict(tmp_dict)
    # for i, node in enumerate(tmp_graph.nodes):
    #     print(f"{i}     {node}")
    #
    # print(f"Head: \n{tmp_graph.head}")
    # print(f"Head child: \n{tmp_graph.head.edges[0]}")


