import networkx as nx
from matplotlib import pyplot as plt
from Benchmark import Warehouse
from networkx.algorithms.shortest_paths.astar import astar_path

from typing import List, Tuple
import time
from collections import deque


def man_dist(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


class GridGraph:
    def __init__(self, grid, strong_orient_root=(9, 4)):
        self._grid = grid
        self.y_len = len(grid)
        self.x_len = len(grid[0])

        # self.G = nx.grid_2d_graph(y_len, x_len)
        self._full_G: nx.Graph = nx.grid_2d_graph(self.y_len, self.x_len)
        self._G: nx.Graph = self._full_G.copy()
        # self._di_G: nx.DiGraph = None

        self._corridor_to_edge_map = {}  # Maps corridor nodes to corresponding edge in G
        self._orig_corridor_to_edge_map = {}
        self._edge_to_corridor_map = {}
        self._orig_edge_to_corridor_map = {}
        self._access_points = {}

        self._invalid_edges = []
        self._invalid_nodes = []
        self._unreachable_nodes = []  # Unreachable 'obstructed' nodes - i.e. start or goal locations that are unreachable

        self._proc_obstacles()
        self._proc_corridors()
        self._G_to_weighted()

        self._G = self._G.to_directed()

        self._G_to_strong_orient(strong_orient_root)
        self._proc_corridor_to_edge_map()

        self._find_unreachable_nodes()

        self.G = self._G

    def get_G(self):
        return self._G

    def get_unreachable_nodes(self):
        return self._unreachable_nodes

    @staticmethod
    def _nodes_along_edge(edge):
        node_1, node_2 = edge
        nodes = []
        if node_1[0] == node_2[0]:  # horizontal
            y = node_1[0]
            if node_1[1] < node_2[1]:  # left to right
                nodes = [(y, x) for x in range(node_1[1] + 1, node_2[1])]
            else:  # right to left
                nodes = [(y, x) for x in range(node_1[1] - 1, node_2[1], -1)]
        elif node_1[1] == node_2[1]:  # Vertical
            x = node_1[1]
            if node_1[0] < node_2[0]:  # top to bottom -> going down
                nodes = [(y, x) for y in range(node_1[0] + 1, node_2[0])]
            else:  # bottom to top -> going up
                nodes = [(y, x) for y in range(node_1[0] - 1, node_2[0], -1)]
        else:
            raise ValueError(f"edge cannot be diagonal. x or y values must match.")
        return nodes

    def _proc_obstacles(self):
        grid = self._grid
        full_G = self._full_G

        # Mark nodes as obstructed or not
        for node in full_G.nodes:
            if grid[node[0]][node[1]] > 0:
                full_G.nodes[node]['obstructed'] = True
            else:
                full_G.nodes[node]['obstructed'] = False

        # Mark edges as obstructed or not
        for edge in full_G.edges:
            if full_G.nodes[edge[0]]['obstructed'] or full_G.nodes[edge[1]]['obstructed']:
                full_G.edges[edge]['obstructed'] = True
            else:
                full_G.edges[edge]['obstructed'] = False

        # Remove obstructed nodes from G
        obstructed_nodes = [node for node, data in full_G.nodes(data=True) if data['obstructed'] is True]
        self._G.remove_nodes_from(obstructed_nodes)

    def _proc_corridors(self):
        G = self._G

        def filter_by_degree(el, min_degree):
            pos, degree = el
            if degree < min_degree:
                return True
            else:
                return False

        corridor_nodes = list(filter(lambda el: filter_by_degree(el, 3), list(G.degree)))
        corridor_nodes = [el[0] for el in corridor_nodes]

        # Above wrongly includes corner nodes as corridor nodes so remove corner nodes from list
        def is_corner(node):
            edge = list(G.edges(node))

            if len(edge) < 2:
                return False

            node1 = edge[0][1]

            node2 = edge[1][1]
            if node1[0] == node2[0] or node1[1] == node2[1]:  # Not corner
                return True
            else:  # Is Corner
                return False

        corridor_nodes = list(filter(is_corner, corridor_nodes))

        # Add corridor edges
        seen = set()
        new_edges = []

        for curr_node in corridor_nodes:
            if curr_node not in seen:
                seen.add(curr_node)
                curr_edge = list(G.edges(curr_node))
                prev_node = curr_edge[0][1]
                next_node = curr_edge[1][1]

                prev_edge = curr_edge
                next_edge = curr_edge

                while prev_node:
                    if prev_node in corridor_nodes:  # and prev_node not in seen:
                        prev_edge = list(G.edges(prev_node))
                        prev_node = prev_edge[0][1]
                        seen.add(prev_node)
                    else:
                        prev_node = None

                while next_node:
                    #     print(next_node)
                    if next_node in corridor_nodes:  # and next_node not in seen:
                        next_edge = list(G.edges(next_node))
                        next_node = next_edge[1][1]
                        seen.add(next_node)
                    else:
                        next_node = None

                new_edge = [(curr_edge[0][0], prev_edge[0][1]), (curr_edge[0][0], next_edge[1][1])]
                new_edges.append((new_edge[0][1], new_edge[1][1]))

        G.remove_nodes_from(corridor_nodes)
        G.add_edges_from(new_edges)

        # _full_G
        full_G = self._full_G
        for node in full_G.nodes:
            if node in corridor_nodes:
                full_G.nodes[node]['corridor'] = True
            else:
                full_G.nodes[node]['corridor'] = False
        pass

    def _proc_corridor_to_edge_map(self):
        for edge in self._G.edges:
            self._edge_to_corridor_map[edge] = []
            # curr_corridor_nodes = []
            node_1, node_2 = edge
            if node_1[0] == node_2[0]:  # horizontal
                # wtf = node_1[1] < node_2[1]
                if node_1[1] < node_2[1]:  # left to right
                    for i in range(node_1[1] + 1, node_2[1]):
                        # self._corridor_to_edge_map[(node_1[0], i)] = edge
                        # self._edge_to_corridor_map[edge].append((node_1[0], i))
                        curr_node = (node_1[0], i)
                        self._corridor_to_edge_map[curr_node] = edge
                        self._edge_to_corridor_map[edge].append(curr_node)
                else:  # right to left
                    for i in range(node_1[1] - 1, node_2[1], -1):
                        # self._corridor_to_edge_map[(node_1[0], i)] = edge
                        # self._edge_to_corridor_map[edge].append((node_1[0], i))
                        curr_node = (node_1[0], i)
                        self._corridor_to_edge_map[curr_node] = edge
                        self._edge_to_corridor_map[edge].append(curr_node)
            else:  # Vertical
                if node_1[0] < node_2[0]:  # top to bottom -> going down
                    for i in range(node_1[0] + 1, node_2[0]):
                        curr_node = (i, node_1[1])
                        self._corridor_to_edge_map[curr_node] = edge
                        self._edge_to_corridor_map[edge].append(curr_node)
                else:  # bottom to top -> going up
                    for i in range(node_1[0] - 1, node_2[0], -1):
                        curr_node = (i, node_1[1])
                        self._corridor_to_edge_map[curr_node] = edge
                        self._edge_to_corridor_map[edge].append(curr_node)

        self._orig_corridor_to_edge_map = self._corridor_to_edge_map.copy()
        self._orig_edge_to_corridor_map = self._edge_to_corridor_map.copy()

    def _G_to_weighted(self):
        for edge in self._G.edges:
            source = edge[0]
            target = edge[1]
            self._G[source][target]['weight'] = man_dist(source, target)

    def _find_unreachable_nodes(self):
        full_G = self._full_G
        G = self._G
        corridor_to_edge_map = self._corridor_to_edge_map

        def is_unreachable(node):
            neighbors_data = [(node, full_G.nodes(data=True)[node]) for node in full_G.neighbors(node)]
            reachable_neighbors = [node_data[0] for node_data in neighbors_data \
                                   if node_data[0] in corridor_to_edge_map or node_data[0] in G.nodes]
            return len(reachable_neighbors) == 0

        obstructed_nodes = [node for node, data in full_G.nodes(data=True) if data['obstructed']==True]
        unreachable_nodes = set(filter(lambda x: is_unreachable(x), obstructed_nodes))
        unreachable_nodes = unreachable_nodes.union(set(self._invalid_nodes))
        for invalid_edge in self._invalid_edges:
            unreachable_nodes = unreachable_nodes.union(set(self._nodes_along_edge(invalid_edge)))

        self._unreachable_nodes = unreachable_nodes

        # raise NotImplementedError

    def _G_to_strong_orient(self, root):
        assert type(self._G) == nx.DiGraph, "self._G is not a DiGraph"

        def path_to_reversed_edges(path):
            edges = []
            for i in range(len(path) - 1):
                edges.append((path[i + 1], path[i]))
            return edges

        orig_graph = self._G
        assert type(orig_graph) == nx.DiGraph
        invalid_nodes = []
        graph = orig_graph.copy()
        explored = set(root)
        processed_edges = set()

        q = deque()
        q.append(root)

        while q:
            curr_node = q.popleft()
            for node in orig_graph.neighbors(curr_node):  # Need to do BFS on original graph
                if node not in explored:
                    # path from root to node
                    path_to_node = []
                    p2n_edges_r = []
                    try:
                        path_to_node = astar_path(graph, root, node, man_dist, weight='weight')
                        p2n_edges_r = path_to_reversed_edges(path_to_node)
                        graph.remove_edges_from(p2n_edges_r)
                    except nx.NetworkXNoPath:
                        invalid_nodes.append(node)
                        print(f"path_to_node for {node} not found")

                    # path from node to root
                    path_to_root = []
                    p2r_edges_r = []
                    if len(path_to_node) > 0:
                        try:
                            path_to_root = astar_path(graph, node, root, man_dist, weight='weight')
                            p2r_edges_r = path_to_reversed_edges(path_to_root)
                            graph.remove_edges_from(p2r_edges_r)
                        except nx.NetworkXNoPath:
                            graph.add_edges_from(p2n_edges_r)
                            invalid_nodes.append(node)
                            print(f"path_to_root for {node} not found")

                    cycle_edges_r = p2n_edges_r + p2r_edges_r
                    for edge in cycle_edges_r:
                        processed_edges.add(edge)
                        processed_edges.add(tuple(reversed(edge)))

                    explored.add(node)
                    q.append(node)

        # Need to deal with edges that are still 2 way
        unprocessed_edges = set(orig_graph.edges) - processed_edges
        edge_seen = set()
        remove_edges = []
        for edge in unprocessed_edges:
            if edge not in edge_seen:
                edge_pair = tuple(reversed(edge))
                edge_seen.add(edge)
                edge_seen.add(edge_pair)
                # Arbitrarily choosing which edge to remove -> Can do smarter way
                remove_edges.append(edge)

        # Some of the edges don't have weights -> Probs cause of above code block
        unweighted_edges = [edge for edge in graph.edges(data="weight", default=-1) if edge[2] < 0]
        for edge in unweighted_edges:
            graph.edges[edge[0], edge[1]]['weight'] = man_dist(edge[0], edge[1])

        graph.remove_edges_from(remove_edges)

        self._invalid_edges = []
        for node in invalid_nodes:
            self._invalid_edges.extend(list(graph.edges(node)))

        # remove invalid nodes
        graph.remove_nodes_from(invalid_nodes)
        self._invalid_nodes = invalid_nodes

        self._G = graph

    def _add_node_on_edge(self, node, edge):
        # edge (node_1, node_2) -> insert node between node_1 and node_2 and create 2 new edges
        node_1 = edge[0]
        node_2 = edge[1]
        self._G.remove_edge(*edge)
        self._G.add_node(node, obstructed=False)
        weighted_edges = [(node_1, node, man_dist(node_1, node)), (node, node_2, man_dist(node, node_2))]
        self._G.add_weighted_edges_from(weighted_edges)

        self._c2e_split(edge, [(node_1, node), (node, node_2)])
        # return weighted_edges

    # If node lies between two edges, remove node and connect edges
    # E.g. remove B from AB and AC
    # A -> B -> C
    # becomes
    # A -> C
    def _remove_node_from_edges(self, node, edges):
        self._G.remove_node(node)
        new_edge = (edges[0][0], edges[1][1])
        weight = man_dist(new_edge[0], new_edge[1])
        self._G.add_weighted_edges_from([(*new_edge, weight)])
        self._c2e_merge(edges)

    def _remove_node_from_edge(self, node):
        self._G.remove_node(node)
        orig_edge = self._orig_corridor_to_edge_map[node]
        nodes_on_orig_edge = self._orig_edge_to_corridor_map[orig_edge]
        prev_node = None
        next_node = None

        for i in range(len(nodes_on_orig_edge)):
            if nodes_on_orig_edge[i] == node:
                if i != 0:
                    prev_node = nodes_on_orig_edge[i-1]
                else:
                    prev_node = orig_edge[0]
                if i < len(nodes_on_orig_edge) - 1:
                    next_node = nodes_on_orig_edge[i+1]
                else:
                    next_node = orig_edge[1]

        edge_1 = [None, node]
        if prev_node in self._corridor_to_edge_map:
            edge_1[0] = self._corridor_to_edge_map[prev_node][0]
        elif prev_node in self._G.nodes:
            edge_1[0] = prev_node
        else:
            raise ValueError(f"{prev_node} not in _corridor_to_edge_map or _G.nodes")
        edge_1 = tuple(edge_1)

        edge_2 = [node, None]
        if next_node in self._corridor_to_edge_map:
            edge_2[1] = self._corridor_to_edge_map[next_node][1]
        elif next_node in self._G.nodes:
            edge_2[1] = next_node
        else:
            raise ValueError(f"{next_node} not in _corridor_to_edge_map or _G.nodes")
        edge_2 = tuple(edge_2)

        new_edge = (edge_1[0], edge_2[1])
        weight = man_dist(new_edge[0], new_edge[1])
        self._G.add_weighted_edges_from([(*new_edge, weight)])
        #
        # edge_1 = (edge[0], node)
        # edge_2 = (node, edge[1])
        self._c2e_merge([edge_1,  edge_2])

    # Split edge in corridor_to_edge_map into two edges
    def _c2e_split(self, old_edge, new_edges):
        assert len(new_edges) == 2, "Length of new_edges must be 2"
        assert new_edges[0][1] == new_edges[1][0], "new_edges must share common vertex"

        mid_node = new_edges[0][1]

        del self._corridor_to_edge_map[mid_node]

        # Could optimise to use 'nodes_along_old_edge' rather than '_nodes_along_edge' call
        nodes_along_old_edge = self._edge_to_corridor_map[old_edge]

        del self._edge_to_corridor_map[old_edge]

        # Should optimise this at some point
        nodes_along_edge_1 = GridGraph._nodes_along_edge(new_edges[0])
        nodes_along_edge_2 = GridGraph._nodes_along_edge(new_edges[1])
        self._edge_to_corridor_map[new_edges[0]] = nodes_along_edge_1
        self._edge_to_corridor_map[new_edges[1]] = nodes_along_edge_2

        for node in nodes_along_edge_1:
            self._corridor_to_edge_map[node] = new_edges[0]

        for node in nodes_along_edge_2:
            self._corridor_to_edge_map[node] = new_edges[1]

    # Merge edges in 'edges' in corridor_to_edge_map
    def _c2e_merge(self, edges: List[Tuple]):
        assert len(edges) == 2, "Length of edges must be 2"
        assert edges[0][1] == edges[1][0], "edges must share common vertex"

        mid_node = edges[0][1]

        new_edge = (edges[0][0], edges[1][1])
        nodes_along_edge_1 = self._edge_to_corridor_map[edges[0]]
        nodes_along_edge_2 = self._edge_to_corridor_map[edges[1]]

        nodes_along_edge = nodes_along_edge_1.copy()
        nodes_along_edge.append(mid_node)
        nodes_along_edge.extend(nodes_along_edge_2)
        self._edge_to_corridor_map[new_edge] = nodes_along_edge

        del self._edge_to_corridor_map[edges[0]]
        del self._edge_to_corridor_map[edges[1]]

        for node in nodes_along_edge:
            self._corridor_to_edge_map[node] = new_edge
        # self._corridor_to_edge_map[mid_node] = new_edge

    # Temporarily add nodes that connect graph to given pos based off neighbours in _full_G
    # TODO - Update corridor to edge map
    def add_access_points(self, node):
        access_points = self._access_points
        access_points[node] = {"nodes":[], "edges":[]}
        self._G.add_node(node, obstructed=True)
        for neighbor in self._full_G.neighbors(node):
            neighbor_data = self._full_G.nodes(data=True)[neighbor]
            is_obstructed = neighbor_data['obstructed']
            if neighbor in self._G.nodes:
                pass
            elif neighbor in self._corridor_to_edge_map:
                edge = self._corridor_to_edge_map[neighbor]
                self._add_node_on_edge(neighbor, edge)
                access_points[node]["nodes"].append(neighbor)
            else:
                continue

            access_points[node]["edges"].extend([(node, neighbor), (neighbor, node)])

            self._G.add_weighted_edges_from([(node, neighbor,  1), (neighbor, node,  1)])

        self._access_points = access_points

    # TODO - Update corridor to edge map
    def remove_access_points(self, node):
        remove_nodes = self._access_points[node]["nodes"]
        for remove_node in remove_nodes:
            # Need to use original corridor_to_edge map
            self._remove_node_from_edge(remove_node)  # , self._orig_corridor_to_edge_map[remove_node])
        self._G.remove_node(node)  # I think this should remove all edges corresponding to node and access points

    def plot_full_G(self, file_name, draw_labels=False):
        full_G = self._full_G
        plt.figure(figsize=(15, 15))
        pos = {(x, y): (y, -x) for x, y in full_G.nodes()}

        # Nodes
        obstructed_nodes = [node for node, data in full_G.nodes(data=True) if data['obstructed'] is True]
        unobstructed_nodes = [node for node, data in full_G.nodes(data=True) if data['obstructed'] is False]
        corridor_nodes = [node for node, data in full_G.nodes(data=True) if data['corridor'] is True]

        # nx.draw(full_G, pos=pos, nodelist=[])
        nx.draw_networkx_nodes(full_G, pos=pos, nodelist=obstructed_nodes, node_size=150, node_color='red')
        nx.draw_networkx_nodes(full_G, pos=pos, nodelist=unobstructed_nodes, node_size=150, node_color='blue')
        nx.draw_networkx_nodes(full_G, pos=pos, nodelist=corridor_nodes, node_size=150, node_color='green')

        # Edges
        obstructed_edges = [(node_1, node_2) for node_1, node_2, data in full_G.edges(data=True) if
                            data['obstructed'] == True]
        unobstructed_edges = [(node_1, node_2) for node_1, node_2, data in full_G.edges(data=True) if
                              data['obstructed'] == False]

        nx.draw_networkx_edges(full_G, pos=pos, edgelist=obstructed_edges,
                               edge_color='red')  # , node_size=150, node_color='red')
        nx.draw_networkx_edges(full_G, pos=pos, edgelist=unobstructed_edges,
                               edge_color='blue')  # , node_size=150, node_color='blue')
        plt.savefig(file_name)

    def plot_G(self, file_name, figsize=(15, 15), draw_weights=False):
        G = self._G
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

# class GridGraph:
#     def __init__(self, grid):
#         self.grid = grid
#         y_len = len(grid)
#         x_len = len(grid[0])
#
#         # self.G = nx.grid_2d_graph(y_len, x_len)
#         self._orig_G = nx.grid_2d_graph(y_len, x_len)
#
#         self.obstacles = []
#         self.obstacles_removed = False
#
#     def find_obstacles(self):
#         grid = self.grid
#         obstacles = []
#         for y in range(len(grid)):
#             for x in range(len(grid[0])):
#                 if grid[y][x]:
#                     obstacles.append((y, x))
#         self.obstacles = obstacles
#         return obstacles
#
#     def remove_obstacles(self):
#         self.obstacles_removed = True
#         obstacles = []
#         if self.obstacles:
#             obstacles = self.obstacles
#         else:
#             obstacles = self.find_obstacles()
#         self.G.remove_nodes_from(obstacles)
#
#     @staticmethod
#     def filter_by_degree(el, min_degree):
#         pos, degree = el
#         if degree < min_degree:
#             return True
#         else:
#             return False
#
#     def is_corner(self, node):
#         edge = list(self.G.edges(node))
#         node1 = edge[0][1]
#
#         if len(edge) == 1:
#             return False
#
#         node2 = edge[1][1]
#         if node1[0] == node2[0] or node1[1] == node2[1]:  # Not corner
#             return True
#         else:  # Is Corner
#             return False
#
#     def convert_to_sparse(self):
#         if not self.obstacles_removed:
#             self.remove_obstacles()
#
#         G = self.G
#         remove_nodes = list(filter(lambda el: GridGraph.filter_by_degree(el, 3), list(G.degree)))
#         remove_nodes = [el[0] for el in remove_nodes]
#         remove_nodes_nc = list(filter(self.is_corner, remove_nodes))
#         # G.remove_nodes_from(remove_nodes_nc)
#         seen = set()
#         new_edges = []
#
#         for curr_node in remove_nodes_nc:
#             if curr_node not in seen:
#                 seen.add(curr_node)
#                 curr_edge = list(G.edges(curr_node))
#                 prev_node = curr_edge[0][1]
#                 next_node = curr_edge[1][1]
#
#                 prev_edge = curr_edge
#                 next_edge = curr_edge
#
#                 while prev_node:
#                     if prev_node in remove_nodes_nc:  # and prev_node not in seen:
#                         prev_edge = list(G.edges(prev_node))
#                         prev_node = prev_edge[0][1]
#                         seen.add(prev_node)
#                     else:
#                         prev_node = None
#
#                 while next_node:
#                     #     print(next_node)
#                     if next_node in remove_nodes_nc:  # and next_node not in seen:
#                         next_edge = list(G.edges(next_node))
#                         next_node = next_edge[1][1]
#                         seen.add(next_node)
#                     else:
#                         next_node = None
#
#                 new_edge = [(curr_edge[0][0], prev_edge[0][1]), (curr_edge[0][0], next_edge[1][1])]
#                 new_edges.append((new_edge[0][1], new_edge[1][1]))
#
#         for edge in new_edges:
#             G.add_edge(edge[0], edge[1])
#         G.remove_nodes_from(remove_nodes_nc)
#         self.G = G
#         return G
#
#     def add_weights_man_dist(self):
#         G = self.G
#         for edge in G.edges:
#             source = edge[0]
#             target = edge[1]
#             G[source][target]['weight'] = man_dist(source, target)
#         self.G = G
#
#     def convert_to_di_graph(self):
#         self.G = self.G.to_directed()
#
#     @staticmethod
#     def path_to_edges(path):
#         edges = []
#         for i in range(len(path) - 1):
#             edges.append((path[i], path[i + 1]))
#         return edges
#
#     @staticmethod
#     def edges_2_way(edges):
#         new_edges = []
#         for edge in edges:
#             new_edges.append(edge)
#             new_edges.append((edge[1], edge[0]))
#         return new_edges
#
#     @staticmethod
#     def get_boundary_nodes(inner_graph, outer_graph):
#         boundary_nodes = set()
#         for edge in outer_graph.edges():
#             if edge[0] in inner_graph.nodes() and edge[1] not in inner_graph.nodes():
#                 boundary_nodes.add(edge[1])
#             elif edge[1] in inner_graph.nodes() and edge[0] not in inner_graph.nodes():
#                 boundary_nodes.add(edge[0])
#         return list(boundary_nodes)
#
#     @staticmethod
#     def get_edges_weights(graph, edges, weight_inc=0):
#         edges_weights = []
#         #     graph_edges = graph.edges(data=True)
#         for edge in edges:
#             weight = graph.edges[edge]['weight'] + weight_inc
#             edges_weights.append((edge[0], edge[1], weight))
#         return edges_weights
#
#     # pass
#     def add_node_along_edge(self, pos1, pos2):
#         pass
#
#     def get_strong_orientation(self, root, weight_inc=0):
#         di_G = self.G.copy()
#         new_G = nx.DiGraph()
#         new_G.add_node(root)
#
#         curr_node = root
#         while curr_node:
#             # Path from node to root #
#             path_nodes = astar.astar_path(di_G, curr_node, root, man_dist, weight='weight')
#
#             path_edges = GridGraph.path_to_edges(path_nodes)
#             path_edges_weights = GridGraph.get_edges_weights(di_G, path_edges, weight_inc=weight_inc)
#             new_G.add_nodes_from(path_nodes)
#             new_G.add_weighted_edges_from(path_edges_weights)
#
#             di_G.remove_edges_from(GridGraph.edges_2_way(path_edges))
#             di_G.add_weighted_edges_from(path_edges_weights, weight='weight')
#
#             # Path from root to node #
#             path_nodes = astar.astar_path(di_G, root, curr_node, man_dist, weight='weight')
#
#             path_edges = GridGraph.path_to_edges(path_nodes)
#             path_edges_weights = GridGraph.get_edges_weights(di_G, path_edges, weight_inc=weight_inc)
#             new_G.add_nodes_from(path_nodes)
#             new_G.add_weighted_edges_from(path_edges_weights)
#
#             di_G.remove_edges_from(GridGraph.edges_2_way(path_edges))
#             di_G.add_weighted_edges_from(path_edges_weights, weight='weight')
#
#             boundary_nodes = GridGraph.get_boundary_nodes(new_G, di_G)
#             #     print(f"Boundary nodes: {boundary_nodes}")
#             if len(boundary_nodes) > 0:
#                 curr_node = boundary_nodes[0]
#             else:
#                 curr_node = None
#
#         return new_G
#
#     def find_path_astar(self, start_pos, goal_pos):
#         G = self.G
#         # start_node = G.nodes[start_pos]
#         # goal_node = G.nodes[goal_pos]
#         return astar.astar_path(self.G, start_pos, goal_pos, heuristic=GridGraph.man_dist_node, weight="weight")
#
#     def save_graph_to_png(self, file_name, figsize=(15, 15), draw_weights=False):
#         G = self.G
#         plt.figure(figsize=figsize)
#         pos = {(x, y): (y, -x) for x, y in G.nodes()}
#         nx.draw(G, pos=pos,
#                 node_color='lightgreen',
#                 with_labels=True,
#                 node_size=200)
#         if draw_weights:
#             # Create edge labels # e[2]['weight']
#             labels = {(e[0], e[1]): e[2]['weight'] for e in G.edges(data=True)}
#             nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#
#         plt.savefig(file_name)
#
#
# def get_strong_oriented_graph(grid):
#     grid_graph = GridGraph(grid)
#     grid_graph.convert_to_sparse()
#     grid_graph.add_weights_man_dist()
#
#     grid_graph.convert_to_di_graph()
#     strong_orient_g = grid_graph.get_strong_orientation((0, 0))
#
#     return strong_orient_g


def example():
    grid = Warehouse.txt_to_grid("map_warehouse_unreachable.txt", use_curr_workspace=True)

    # start = time.time()
    grid_graph = GridGraph(grid)

    grid_graph.plot_full_G("full_G.png")
    grid_graph.plot_G("G.png", draw_weights=True)
    # grid_graph.convert_to_sparse()
    # grid_graph.add_weights_man_dist()
    #
    # grid_graph.convert_to_di_graph()
    # strong_orient_g = grid_graph.get_strong_orientation((0, 0))

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

    # grid_graph.save_graph_to_png("grid_graph.png", draw_weights=True)


if __name__ == "__main__":
    example()
