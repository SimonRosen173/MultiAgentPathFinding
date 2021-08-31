# Node for space-time graph
class Node:
    def __init__(self, parent, x, y, t):
        self.x = x
        self.y = y
        self.t = t
        self.parent = parent
        self.edges = []  # list of tuples of edge length and node
        self.children = []

    def add_node(self, edge_len, node_ind):
        self.edges.append((edge_len, node_ind))

    def is_pos_equal(self, other):
        return self.x == other.x and self.y == other.y

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.t == other.t:
            return True
        else:
            return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Node({self.x},{self.y},{self.t})"
        # return f"NODE ({self.x},{self.y},{self.t}) : parent = {self.parent}, edges = {str(self.edges)}"


class NodeAlt:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.edges = []

    def add_edge(self, neighbour: Node, weight: int):
        self.edges.append((neighbour, weight))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"NodeAlt({self.x},{self.y})"


class SpaceTimeNode(Node):
    def __init__(self, parent, x, y):
        super().__init__(self, parent, x, y)


class Graph:
    def __init__(self, head: Node, x_size: int, y_size: int):
        self.head: Node = head
        self.x_size = x_size
        self.y_size = y_size
        self.nodes = []
        self.edges = []  # Not sure if required

    def init_from_edge_dict(self, node_dict):
        self.nodes = [None]*len(node_dict)  # No idea if this leads to speed ups
        for i, el_ind in enumerate(node_dict):
            el = node_dict[el_ind]
            tmp_node = Node(None, el[0][0], el[0][1], el[0][2])
            self.nodes[i] = tmp_node

        for i, el_ind in enumerate(node_dict):
            el = node_dict[el_ind]
            curr_node: Node = self.nodes[i]  # Typing for IDE
            neighbour_inds = el[1]
            for neighbour_ind in neighbour_inds:
                curr_node.edges.append((1, neighbour_ind))

            if i == 0:
                self.head = curr_node

    ##################
    # Visualisations #
    ##################
    def graph_to_dot_file(self, file_name):
        pass

    # Grid for each time step -> How useful?
    def graph_to_grids(self):
        pass


if __name__ == "__main__":
    tmp_dict = {
        0: ([0, 0, 0], [1, 2, 3]),
        1: ([1, 2, 0], [2]),
        2: ([3, 4, 0], [1, 2]),
        3: ([5, 6, 0], [0, 2])
    }
    tmp_graph = Graph(None, 10, 10)
    tmp_graph.init_from_edge_dict(tmp_dict)
    for i, node in enumerate(tmp_graph.nodes):
        print(f"{i}     {node}")

    print(f"Head: \n{tmp_graph.head}")
    print(f"Head child: \n{tmp_graph.head.edges[0]}")


