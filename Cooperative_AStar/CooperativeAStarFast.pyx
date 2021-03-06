# print("Hello World")

from heapq import heappush, heappop
from itertools import count

import numpy as np
from cpython cimport array

from more_itertools import flatten
import networkx as nx
from Benchmark import Warehouse
from GlobalObjs.GraphNX import GridGraph, plot_graph
from Visualisations.Vis import VisGrid
from MAPD.TaskAssigner import TaskAssigner
from typing import Dict, List, Tuple, Set, Optional
from numba import njit, typed
import time

cimport numpy as np
DTYPE = int  # np.int
ctypedef np.int_t DTYPE_t

cdef man_dist(int x_1, int x_2, int y_1, int y_2):
    return abs(x_1 - x_2) + abs(y_1 - y_2)


cdef is_neighbour_valid(np.ndarray[DTYPE_t, ndim=2] grid, int x, int y):
    cdef int max_y = grid.shape[0] - 1
    cdef int max_x = grid.shape[1] - 1

    # cdef bint is_valid = True
    return 0 <= x <= max_x and 0<= y <= max_y

    # return  is_valid


def cooperative_astar_path(np.ndarray[DTYPE_t, ndim=2] grid,
                          tuple source, tuple target,
                          # int target_x, int target_y,
                          set resv_tbl, set resv_locs,
                          int start_t, int cutoff_t = 500
                          ):
    # print(source_x, source_y)

    cdef list path = []

    # cdef cutoff_t = 500

    c = count()

    q = [(0, next(c), (source, start_t), 0, None)]  # _, __, curnode, t, dist, parent

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    cdef dict enqueued = {}
    # Maps explored nodes to parent closest to the source.
    cdef dict explored = {}
    cdef bint path_found = False

    # tmp = (source, start_t) in resv_tbl
    # cdef list target = [target_x, target_y]
    # cdef list source = [source_x, source_y]
    cdef int x = 0
    cdef int y = 0
    cdef int t = 0
    cdef tuple curnode = None
    cdef list neighbours = []
    cdef int max_y = grid.shape[0] - 1
    cdef int max_x = grid.shape[1] - 1
    cdef int curr_x = 0
    cdef int curr_y = 0

    assert 0 <= source[0] <= max_x and 0<= source[1] <= max_y, f"Source {source} not valid"
    assert 0 <= target[0] <= max_x and 0<= target[1] <= max_y, f"Target {target} not valid"

    while q and not path_found:
        # Pop the smallest item from queue.
        tmp, __, curnode, dist, parent = heappop(q)

        # print(tmp, curnode)

        # print(curnode, target)
        # If target found
        if curnode[0][0] == target[0] and curnode[0][1] == target[1]:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]

            path.reverse()
            return path

        #
        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        next_t = curnode[1] + 1

        x = curnode[0][0]
        y = curnode[0][1]

        # Neighbors and wait action
        neighbours = []
        # Left
        if 0 <= x-1 <= max_x and 0<= y <= max_y:
            neighbours.append(((x-1, y), next_t))
        # Right
        if 0 <= x+1 <= max_x and 0<= y <= max_y:
            neighbours.append(((x+1, y), next_t))
        # Up
        if 0 <= x <= max_x and 0<= y-1 <= max_y:
            neighbours.append(((x, y-1), next_t))
        # Down
        if 0 <= x <= max_x and 0<= y+1 <= max_y:
            neighbours.append(((x, y+1), next_t))
        # Wait
        neighbours.append(((x,y), next_t))

        for neighbour, curr_t in neighbours:
            # Skip neighbor if obstructed and not source or target
            curr_x = neighbour[0]
            curr_y = neighbour[1]

            if grid[curr_y, curr_x] != 0 and not (neighbour[0] == target[0] and neighbour[1] == target[1]) \
                    and not (neighbour[0] == source[0] and neighbour[1] == source[1]):
                # print("Obstacle at ", curr_x, curr_y)
                continue

            # curr_t = curnode[1] + curr_weight
            ncost = dist + 1 # weight(curnode[0], neighbor, w)
            # curr_t = ncost

            if curr_t - start_t > cutoff_t:
                # raise Exception("Maximum number of timesteps reached.")
                print("Maximum number of timesteps reached.")
                return []

            if (neighbour, curr_t) not in resv_tbl and neighbour not in resv_locs:
                # I.e. if neighbor is a stationary agent
                # if neighbor in resv_locs:
                #     intervals = resv_locs[neighbor]  # Intervals when neighbor is stationary
                #     is_neighbor_valid=True
                #     for interval in intervals:
                #         if interval[0] <= curr_t <= interval[1]:
                #             is_neighbor_valid = False
                #             continue
                #
                #     if not is_neighbor_valid:
                #         continue

                if (neighbour, curr_t) in enqueued:
                    qcost, h = enqueued[(neighbour, curr_t)]
                    # if qcost <= ncost, a less costly path from the
                    # neighbor to the source was already determined.
                    # Therefore, we won't attempt to push this neighbor
                    # to the queue
                    if qcost <= ncost:
                        continue
                else:
                    h = man_dist(neighbour[0], neighbour[1], target[0], target[1])

                enqueued[(neighbour, curr_t)] = ncost, h

                heappush(q, (ncost + h, next(c), (neighbour, curr_t), ncost, curnode))

    print("How?")

# def cooperative_astar_path_old(G,
#                            sources: List[Tuple[int, int]], targets: List[Tuple[int, int]],
#                            resv_tbl: Optional[Set[Tuple[Tuple[int, int], int]]] = None,
#                            # resv_locs - locations reserved from some timestep onwards
#                            resv_locs: Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]] = None,
#                            start_t=0) -> Tuple[List[Optional[Optional[List[Tuple[Tuple[int, int], int]]]]], Set]:
#     assert len(sources) == len(targets), "Length of sources must equal length of targets"
#
#     # orig_resv_tbl = resv_tbl.copy()
#     resv_tbl = resv_tbl.copy()
#     cutoff_t = 500
#
#     for source, target in zip(sources, targets):
#         if source not in G or target not in G:
#             msg = f"Either source {source} or target {target} is not in G"
#             raise nx.NodeNotFound(msg)
#
#     no_agents = len(sources)
#     paths: List[Optional[Optional[List[Tuple[Tuple[int, int], int]]]]] = []
#     if resv_tbl is None:
#         resv_tbl = set()
#
#     if resv_locs is None:
#         resv_locs = {}
#
#     max_t = start_t
#
#     push = heappush
#     pop = heappop
#     weight = 1
#
#     for agent_ind, (source, target) in enumerate(zip(sources, targets)):
#
#         # The queue stores priority, node, cost to reach, and parent.
#         # Uses Python heapq to keep in priority order.
#         # Add a counter to the queue to prevent the underlying heap from
#         # attempting to compare the nodes themselves. The hash breaks ties in the
#         # priority and is guaranteed unique for all nodes in the graph.
#         # curnode consists of tuple of current node and time spent at node
#         # e.g. ((0,0), 1) signifies agent has spent 1 timestep at (0, 0)
#         c = count()
#         queue = [(0, next(c), (source, start_t), start_t, 0, None)]  # _, __, curnode, t, dist, parent
#
#         # Maps enqueued nodes to distance of discovered paths and the
#         # computed heuristics to target. We avoid computing the heuristics
#         # more than once and inserting the node into the queue too many times.
#         enqueued = {}
#         # Maps explored nodes to parent closest to the source.
#         explored = {}
#         path_found = False
#
#         tmp = (source, start_t) in resv_tbl
#
#         while queue and not path_found:
#             # Pop the smallest item from queue.
#             _, __, curnode, t, dist, parent = pop(queue)
#             # if agent_ind == 1:
#             #     print(curnode)
#
#             # If target found
#             if curnode[0] == target:
#                 path = [curnode]
#                 node = parent
#                 while node is not None:
#                     path.append(node)
#                     node = explored[node]
#
#                 path.reverse()
#
#                 for node in path:
#                     resv_tbl.add(node)
#                     resv_tbl.add((node[0], node[1]+1))
#
#                 max_t = max(path[-1][1], max_t)
#                 path_found = True
#                 paths.append(path)
#                 break
#
#             if curnode in explored:
#                 # Do not override the parent of starting node
#                 if explored[curnode] is None:
#                     continue
#
#                 # Skip bad paths that were enqueued before finding a better one
#                 qcost, h = enqueued[curnode]
#                 if qcost < dist:
#                     continue
#
#             explored[curnode] = parent
#
#             next_t = curnode[1] + 1  # Probs where the issue is coming from
#
#             # Allow for wait action where agent does not move
#             neighbors = [(node, next_t) for node in G[curnode[0]].items()]
#             tmp = neighbors + [((curnode[0], {'weight': None}), next_t)]
#
#             for (neighbor, w), curr_t in tmp:
#                 # Skip neighbor if obstructed and not source or target
#                 if G.nodes(data="obstructed")[neighbor] is True and neighbor != target and neighbor != source:
#                     continue
#
#                 # + 1 to account for moving out of current node
#                 # if (neighbor, t + weight(curnode[0], neighbor, w)) not in resv_tbl:
#                 if curnode[0] == neighbor:
#                     curr_weight = 1
#                 else:
#                     curr_weight = weight(curnode[0], neighbor, w)
#
#                 # curr_t = curnode[1] + curr_weight
#                 ncost = dist + curr_weight # weight(curnode[0], neighbor, w)
#                 # curr_t = ncost
#
#                 if curr_t - start_t > cutoff_t:
#                     # raise Exception("Maximum number of timesteps reached.")
#                     print("Maximum number of timesteps reached.")
#                     return None, None
#
#                 if (neighbor, curr_t) not in resv_tbl:
#                     # I.e. if neighbor is a stationary agent
#                     if neighbor in resv_locs:
#                         intervals = resv_locs[neighbor]  # Intervals when neighbor is stationary
#                         is_neighbor_valid=True
#                         for interval in intervals:
#                             if interval[0] <= curr_t <= interval[1]:
#                                 is_neighbor_valid = False
#                                 continue
#
#                         if not is_neighbor_valid:
#                             continue
#
#                     if (neighbor, curr_t) in enqueued:
#                         qcost, h = enqueued[(neighbor, curr_t)]
#                         # if qcost <= ncost, a less costly path from the
#                         # neighbor to the source was already determined.
#                         # Therefore, we won't attempt to push this neighbor
#                         # to the queue
#                         if qcost <= ncost:
#                             continue
#                     else:
#                         h = 1
#                     enqueued[(neighbor, curr_t)] = ncost, h
#
#                     push(queue, (ncost + h, next(c), (neighbor, curr_t), t + curr_weight, ncost, curnode))
#
#         if not path_found:
#             # print(f"Node {target} not reachable from {source} for agent {agent_ind}")
#             # # Why?
#             # resv_tbl_list = list(orig_resv_tbl)
#             # resv_tbl_pos = {el[0]: el[1] for el in resv_tbl_list}
#             # if source in resv_tbl_pos:
#             #     print(f"Source {source} is in resv_tbl for ({source}, {resv_tbl_pos[source]})")
#             # if target in resv_tbl_pos:
#             #     print(f"Target {target} is in resv_tbl for ({target}), {resv_tbl_pos[target]})")
#
#             paths.append(None)
#             # raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} for agent {agent_ind}")
#
#     return paths, resv_tbl