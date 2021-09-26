import networkx as nx
from matplotlib import pyplot as plt
from Benchmark import Warehouse
from networkx.algorithms.shortest_paths import astar
import time


def man_dist(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


class GridGraph:
    def __init__(self, grid):
        self.grid = grid
        y_len = len(grid)
        x_len = len(grid[0])

        self.G = nx.grid_2d_graph(y_len, x_len)
        self.obstacles = []
        self.obstacles_removed = False

    def find_obstacles(self):
        grid = self.grid
        obstacles = []
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x]:
                    obstacles.append((y, x))
        self.obstacles = obstacles
        return obstacles

    def remove_obstacles(self):
        self.obstacles_removed = True
        obstacles = []
        if self.obstacles:
            obstacles = self.obstacles
        else:
            obstacles = self.find_obstacles()
        self.G.remove_nodes_from(obstacles)

    @staticmethod
    def filter_by_degree(el, min_degree):
        pos, degree = el
        if degree < min_degree:
            return True
        else:
            return False

    def is_corner(self, node):
        edge = list(self.G.edges(node))
        node1 = edge[0][1]

        if len(edge) == 1:
            return False

        node2 = edge[1][1]
        if node1[0] == node2[0] or node1[1] == node2[1]:  # Not corner
            return True
        else:  # Is Corner
            return False

    def convert_to_sparse(self):
        if not self.obstacles_removed:
            self.remove_obstacles()

        G = self.G
        remove_nodes = list(filter(lambda el: GridGraph.filter_by_degree(el, 3), list(G.degree)))
        remove_nodes = [el[0] for el in remove_nodes]
        remove_nodes_nc = list(filter(self.is_corner, remove_nodes))
        # G.remove_nodes_from(remove_nodes_nc)
        seen = set()
        new_edges = []

        for curr_node in remove_nodes_nc:
            if curr_node not in seen:
                seen.add(curr_node)
                curr_edge = list(G.edges(curr_node))
                prev_node = curr_edge[0][1]
                next_node = curr_edge[1][1]

                prev_edge = curr_edge
                next_edge = curr_edge

                while prev_node:
                    if prev_node in remove_nodes_nc:  # and prev_node not in seen:
                        prev_edge = list(G.edges(prev_node))
                        prev_node = prev_edge[0][1]
                        seen.add(prev_node)
                    else:
                        prev_node = None

                while next_node:
                    #     print(next_node)
                    if next_node in remove_nodes_nc:  # and next_node not in seen:
                        next_edge = list(G.edges(next_node))
                        next_node = next_edge[1][1]
                        seen.add(next_node)
                    else:
                        next_node = None

                new_edge = [(curr_edge[0][0], prev_edge[0][1]), (curr_edge[0][0], next_edge[1][1])]
                new_edges.append((new_edge[0][1], new_edge[1][1]))

        for edge in new_edges:
            G.add_edge(edge[0], edge[1])
        G.remove_nodes_from(remove_nodes_nc)
        self.G = G
        return G

    def add_weights_man_dist(self):
        G = self.G
        for edge in G.edges:
            source = edge[0]
            target = edge[1]
            G[source][target]['weight'] = man_dist(source, target)
        self.G = G

    def convert_to_di_graph(self):
        self.G = self.G.to_directed()

    @staticmethod
    def path_to_edges(path):
        edges = []
        for i in range(len(path) - 1):
            edges.append((path[i], path[i + 1]))
        return edges

    @staticmethod
    def edges_2_way(edges):
        new_edges = []
        for edge in edges:
            new_edges.append(edge)
            new_edges.append((edge[1], edge[0]))
        return new_edges

    @staticmethod
    def get_boundary_nodes(inner_graph, outer_graph):
        boundary_nodes = set()
        for edge in outer_graph.edges():
            if edge[0] in inner_graph.nodes() and edge[1] not in inner_graph.nodes():
                boundary_nodes.add(edge[1])
            elif edge[1] in inner_graph.nodes() and edge[0] not in inner_graph.nodes():
                boundary_nodes.add(edge[0])
        return list(boundary_nodes)

    @staticmethod
    def get_edges_weights(graph, edges, weight_inc=0):
        edges_weights = []
        #     graph_edges = graph.edges(data=True)
        for edge in edges:
            weight = graph.edges[edge]['weight'] + weight_inc
            edges_weights.append((edge[0], edge[1], weight))
        return edges_weights

    # pass
    def add_node_along_edge(self, pos1, pos2):
        pass

    def get_strong_orientation(self, root, weight_inc=0):
        di_G = self.G.copy()
        new_G = nx.DiGraph()
        new_G.add_node(root)

        curr_node = root
        while curr_node:
            # Path from node to root #
            path_nodes = astar.astar_path(di_G, curr_node, root, man_dist, weight='weight')

            path_edges = GridGraph.path_to_edges(path_nodes)
            path_edges_weights = GridGraph.get_edges_weights(di_G, path_edges, weight_inc=weight_inc)
            new_G.add_nodes_from(path_nodes)
            new_G.add_weighted_edges_from(path_edges_weights)

            di_G.remove_edges_from(GridGraph.edges_2_way(path_edges))
            di_G.add_weighted_edges_from(path_edges_weights, weight='weight')

            # Path from root to node #
            path_nodes = astar.astar_path(di_G, root, curr_node, man_dist, weight='weight')

            path_edges = GridGraph.path_to_edges(path_nodes)
            path_edges_weights = GridGraph.get_edges_weights(di_G, path_edges, weight_inc=weight_inc)
            new_G.add_nodes_from(path_nodes)
            new_G.add_weighted_edges_from(path_edges_weights)

            di_G.remove_edges_from(GridGraph.edges_2_way(path_edges))
            di_G.add_weighted_edges_from(path_edges_weights, weight='weight')

            boundary_nodes = GridGraph.get_boundary_nodes(new_G, di_G)
            #     print(f"Boundary nodes: {boundary_nodes}")
            if len(boundary_nodes) > 0:
                curr_node = boundary_nodes[0]
            else:
                curr_node = None

        return new_G

    def find_path_astar(self, start_pos, goal_pos):
        G = self.G
        # start_node = G.nodes[start_pos]
        # goal_node = G.nodes[goal_pos]
        return astar.astar_path(self.G, start_pos, goal_pos, heuristic=GridGraph.man_dist_node, weight="weight")

    def save_graph_to_png(self, file_name, figsize=(15, 15), draw_weights=False):
        G = self.G
        plt.figure(figsize=figsize)
        pos = {(x, y): (y, -x) for x, y in G.nodes()}
        nx.draw(G, pos=pos,
                node_color='lightgreen',
                with_labels=True,
                node_size=200)
        if draw_weights:
            # Create edge labels # e[2]['weight']
            labels = {(e[0], e[1]): e[2]['weight'] for e in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        plt.savefig(file_name)


def get_strong_oriented_graph(grid):
    grid_graph = GridGraph(grid)
    grid_graph.convert_to_sparse()
    grid_graph.add_weights_man_dist()

    grid_graph.convert_to_di_graph()
    strong_orient_g = grid_graph.get_strong_orientation((0, 0))

    return strong_orient_g


def example():
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True)

    start = time.time()
    grid_graph = GridGraph(grid)
    grid_graph.convert_to_sparse()
    grid_graph.add_weights_man_dist()

    grid_graph.convert_to_di_graph()
    strong_orient_g = grid_graph.get_strong_orientation((0, 0))

    # grid_graph.convert_to_di_graph()
    # end = time.time()
    # print(f"Time elapsed: {end - start}")
    #
    # start = time.time()
    # start_pos = (0, 0)
    # goal_pos = (18, 37)
    # path = grid_graph.find_path_astar(start_pos, goal_pos)
    # end = time.time()
    # print(f"Time elapsed: {end - start}")
    # print(path)

    grid_graph.save_graph_to_png("grid_graph.png", draw_weights=True)


if __name__ == "__main__":
    example()
