import heapq
from GlobalObjs.Graph import Node
import os
from Benchmark import Warehouse
from GlobalObjs.Graph import Graph


class BFS:
    def __init__(self):
        pass

    @staticmethod
    def man_dist(node1, node2):
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)

    @staticmethod
    def find_path(start_node: Node, goal_node: Node):
        visited = {(start_node.x, start_node.y)}
        q = [(BFS.man_dist(start_node, goal_node), start_node)]

        while q:
            curr_cost, curr_node = heapq.heappop(q)
            for edge in curr_node.edges:
                neighbour, weight = edge
                if (neighbour.x, neighbour.y) not in visited:
                    neighbour.parent = curr_node
                    if neighbour == goal_node:
                        return neighbour
                    else:
                        visited.add((neighbour.x, neighbour.y))
                        heapq.heappush(q, (new_cost * -1, neighbour))
        return None


def example():
    workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
    # print(workspace_path)
    grid = Warehouse.txt_to_grid(workspace_path + "/Benchmark/maps/map_warehouse.txt", simple_layout=True)
    nodes = Graph.create_from_grid_dense(grid)
    # for node in nodes:
    #     print(nodes[node])
    # print(nodes)
    # print(list(nodes.keys()))
    start = nodes[list(nodes.keys())[0]]
    goal = nodes[list(nodes.keys())[-1]]
    print(f"start - {start}\ngoal - {goal}\n")

    path = BFS.find_path(start, goal)
    print(path)
    # node1 = Node(None, 0, 0)
    # node2 = Node(None, 1, 1)
    # node3 = Node(None, 1, 2)
    # node4 = Node(None, 3, 3)


if __name__ == "__main__":
    example()
