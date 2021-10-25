from heapq import heappush, heappop
from itertools import count

import numpy as np
from more_itertools import flatten
import networkx as nx
from Benchmark import Warehouse
from GlobalObjs.GraphNX import GridGraph, plot_graph
from Visualisations.Vis import VisGrid
from MAPD.TaskAssigner import TaskAssigner
from typing import Dict, List, Tuple, Set, Optional
from numba import njit, typed
import time


def man_dist(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


def _weight_function(G, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def cooperative_astar_path(G, sources: List[Tuple[int, int]], targets: List[Tuple[int, int]], heuristic=man_dist, weight="weight",
                           resv_tbl: Optional[Set[Tuple[Tuple[int, int], int]]] = None,
                           # resv_locs - locations reserved from some timestep onwards
                           resv_locs: Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]] = None,
                           start_t=0) -> Tuple[List[Optional[Optional[List[Tuple[Tuple[int, int], int]]]]], Set]:
    assert len(sources) == len(targets), "Length of sources must equal length of targets"

    orig_resv_tbl = resv_tbl.copy()
    resv_tbl = resv_tbl.copy()
    cutoff_t = 500

    for source, target in zip(sources, targets):
        if source not in G or target not in G:
            msg = f"Either source {source} or target {target} is not in G"
            raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    no_agents = len(sources)
    paths: List[Optional[Optional[List[Tuple[Tuple[int, int], int]]]]] = []
    if resv_tbl is None:
        resv_tbl = set()

    if resv_locs is None:
        resv_locs = {}

    max_t = start_t

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    for agent_ind, (source, target) in enumerate(zip(sources, targets)):

        # The queue stores priority, node, cost to reach, and parent.
        # Uses Python heapq to keep in priority order.
        # Add a counter to the queue to prevent the underlying heap from
        # attempting to compare the nodes themselves. The hash breaks ties in the
        # priority and is guaranteed unique for all nodes in the graph.
        # curnode consists of tuple of current node and time spent at node
        # e.g. ((0,0), 1) signifies agent has spent 1 timestep at (0, 0)
        c = count()
        queue = [(0, next(c), (source, start_t), start_t, 0, None)]  # _, __, curnode, t, dist, parent

        # Maps enqueued nodes to distance of discovered paths and the
        # computed heuristics to target. We avoid computing the heuristics
        # more than once and inserting the node into the queue too many times.
        enqueued = {}
        # Maps explored nodes to parent closest to the source.
        explored = {}
        path_found = False

        tmp = (source, start_t) in resv_tbl

        while queue and not path_found:
            # Pop the smallest item from queue.
            _, __, curnode, t, dist, parent = pop(queue)
            # if agent_ind == 1:
            #     print(curnode)

            # If target found
            if curnode[0] == target:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]

                path.reverse()

                for node in path:
                    resv_tbl.add(node)
                    resv_tbl.add((node[0], node[1]+1))

                max_t = max(path[-1][1], max_t)
                path_found = True
                paths.append(path)
                break

            if curnode in explored:
                # Do not override the parent of starting node
                if explored[curnode] is None:
                    continue

                # Skip bad paths that were enqueued before finding a better one
                qcost, h = enqueued[curnode]
                if qcost < dist:
                    continue

            explored[curnode] = parent

            next_t = curnode[1] + 1  # Probs where the issue is coming from

            # Allow for wait action where agent does not move
            neighbors = [(node, next_t) for node in G[curnode[0]].items()]
            tmp = neighbors + [((curnode[0], {'weight': None}), next_t)]

            for (neighbor, w), curr_t in tmp:
                # Skip neighbor if obstructed and not source or target
                if G.nodes(data="obstructed")[neighbor] is True and neighbor != target and neighbor != source:
                    continue

                # + 1 to account for moving out of current node
                # if (neighbor, t + weight(curnode[0], neighbor, w)) not in resv_tbl:
                if curnode[0] == neighbor:
                    curr_weight = 1
                else:
                    curr_weight = weight(curnode[0], neighbor, w)

                # curr_t = curnode[1] + curr_weight
                ncost = dist + curr_weight # weight(curnode[0], neighbor, w)
                # curr_t = ncost

                if curr_t - start_t > cutoff_t:
                    # raise Exception("Maximum number of timesteps reached.")
                    print("Maximum number of timesteps reached.")
                    return None, None

                if (neighbor, curr_t) not in resv_tbl:
                    # I.e. if neighbor is a stationary agent
                    if neighbor in resv_locs:
                        intervals = resv_locs[neighbor]  # Intervals when neighbor is stationary
                        is_neighbor_valid=True
                        for interval in intervals:
                            if interval[0] <= curr_t <= interval[1]:
                                is_neighbor_valid = False
                                continue

                        if not is_neighbor_valid:
                            continue

                    if (neighbor, curr_t) in enqueued:
                        qcost, h = enqueued[(neighbor, curr_t)]
                        # if qcost <= ncost, a less costly path from the
                        # neighbor to the source was already determined.
                        # Therefore, we won't attempt to push this neighbor
                        # to the queue
                        if qcost <= ncost:
                            continue
                    else:
                        h = heuristic(neighbor, target)
                    enqueued[(neighbor, curr_t)] = ncost, h

                    push(queue, (ncost + h, next(c), (neighbor, curr_t), t + curr_weight, ncost, curnode))

        if not path_found:
            print(f"Node {target} not reachable from {source} for agent {agent_ind}")
            # Why?
            resv_tbl_list = list(orig_resv_tbl)
            resv_tbl_pos = {el[0]: el[1] for el in resv_tbl_list}
            if source in resv_tbl_pos:
                print(f"Source {source} is in resv_tbl for ({source}, {resv_tbl_pos[source]})")
            if target in resv_tbl_pos:
                print(f"Target {target} is in resv_tbl for ({target}), {resv_tbl_pos[target]})")

            paths.append(None)
            # raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} for agent {agent_ind}")

    return paths, resv_tbl


@njit
def man_dist_njit(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


@njit
def cooperative_astar_path_njit(grid: np.ndarray,
                                source: Tuple[int, int], target: Tuple[int, int],
                                resv_tbl: Set[Tuple[int, int]],
                                resv_locs: Set[int],
                                start_t: int, cutoff_t: int):
    # for source, target in zip(sources, targets):
    #     if source not in G or target not in G:
    #         msg = f"Either source {source} or target {target} is not in G"
    #         raise nx.NodeNotFound(msg)

    # no_agents = len(sources)
    # path: List[Tuple[Tuple[int, int], int]] = [((-1, -1), -1)]
    # path.pop(0)
    #
    # max_t = start_t
    #
    # push = heappush
    # pop = heappop
    #
    # # The queue stores priority, node, cost to reach, and parent.
    # # Uses Python heapq to keep in priority order.
    # # Add a counter to the queue to prevent the underlying heap from
    # # attempting to compare the nodes themselves. The hash breaks ties in the
    # # priority and is guaranteed unique for all nodes in the graph.
    # # curnode consists of tuple of current node and time spent at node
    # # e.g. ((0,0), 1) signifies agent has spent 1 timestep at (0, 0)
    # c = 0
    # queue = [(0, c + 1, source, start_t, 0, -1)]  # _, __, curnode, t, dist, parent
    #
    # # Maps enqueued nodes to distance of discovered paths and the
    # # computed heuristics to target. We avoid computing the heuristics
    # # more than once and inserting the node into the queue too many times.
    # enqueued = {(-1,-1): (-1, -1)}
    # # Maps explored nodes to parent closest to the source.
    # explored = {(-1,-1): (-1, -1)}
    # path_found = False
    #
    # while queue and not path_found:
    #     # Pop the smallest item from queue.
    #     _, __, curnode, t, dist, parent = pop(queue)
    #     # if agent_ind == 1:
    #     #     print(curnode)
    #
    #     # If target found
    #     if curnode[0] == target:
    #         path = [(curnode, -1)]
    #         node = (parent, -1)
    #         while node is not ((-1, -1), -1):
    #             path.append(node)
    #             node = explored[node]
    #
    #         path.reverse()
    #         for i, node in enumerate(path):
    #             path[i] =
    #             pass
    #     #
    #     #     for node in path:
    #     #         resv_tbl.add(node)
    #     #         resv_tbl.add((node[0], node[1]+1))
    #     #
    #     #     max_t = max(path[-1][1], max_t)
    #     #     path_found = True
    #     #     paths.append(path)
    #     #     break
    #     #
    #     # if curnode in explored:
    #     #     # Do not override the parent of starting node
    #     #     if explored[curnode] is None:
    #     #         continue
    #     #
    #     #     # Skip bad paths that were enqueued before finding a better one
    #     #     qcost, h = enqueued[curnode]
    #     #     if qcost < dist:
    #     #         continue
    #     #
    #     # explored[curnode] = parent
    #     #
    #     # next_t = curnode[1] + 1  # Probs where the issue is coming from
    #     #
    #     # # Allow for wait action where agent does not move
    #     # neighbors = [(node, next_t) for node in G[curnode[0]].items()]
    #     # tmp = neighbors + [((curnode[0], {'weight': None}), next_t)]
    #     #
    #     # for (neighbor, w), curr_t in tmp:
    #     #     # Skip neighbor if obstructed and not source or target
    #     #     if G.nodes(data="obstructed")[neighbor] is True and neighbor != target and neighbor != source:
    #     #         continue
    #     #
    #     #     # + 1 to account for moving out of current node
    #     #     # if (neighbor, t + weight(curnode[0], neighbor, w)) not in resv_tbl:
    #     #     if curnode[0] == neighbor:
    #     #         curr_weight = 1
    #     #     else:
    #     #         curr_weight = 1
    #     #
    #     #     # curr_t = curnode[1] + curr_weight
    #     #     ncost = dist + curr_weight # weight(curnode[0], neighbor, w)
    #     #     # curr_t = ncost
    #     #
    #     #     if curr_t - start_t > cutoff_t:
    #     #         # raise Exception("Maximum number of timesteps reached.")
    #     #         print("Maximum number of timesteps reached.")
    #     #         return None, None
    #     #
    #     #     if (neighbor, curr_t) not in resv_tbl:
    #     #         # I.e. if neighbor is a stationary agent
    #     #         if neighbor in resv_locs:
    #     #             intervals = resv_locs[neighbor]  # Intervals when neighbor is stationary
    #     #             is_neighbor_valid=True
    #     #             for interval in intervals:
    #     #                 if interval[0] <= curr_t <= interval[1]:
    #     #                     is_neighbor_valid = False
    #     #                     continue
    #     #
    #     #             if not is_neighbor_valid:
    #     #                 continue
    #     #
    #     #         if (neighbor, curr_t) in enqueued:
    #     #             qcost, h = enqueued[(neighbor, curr_t)]
    #     #             # if qcost <= ncost, a less costly path from the
    #     #             # neighbor to the source was already determined.
    #     #             # Therefore, we won't attempt to push this neighbor
    #     #             # to the queue
    #     #             if qcost <= ncost:
    #     #                 continue
    #     #         else:
    #     #             h = 1
    #     #         enqueued[(neighbor, curr_t)] = ncost, h
    #     #
    #     #         push(queue, (ncost + h, next(c), (neighbor, curr_t), t + curr_weight, ncost, curnode))

    return 1  # paths, resv_tbl


@njit
def test_njit_fn(resv_tbl, resv_locs):
    return [(1,1), (2,2), (3,3)]


def cooperative_astar_path_fast(grid,
                                sources: List[Tuple[int, int]], targets: List[Tuple[int, int]],
                                resv_tbl: Optional[Set[Tuple[Tuple[int, int], int]]] = None,
                                # resv_locs - locations reserved from some timestep onwards
                                resv_locs: Optional[Set[Tuple[int, int]]] = None,
                                start_t=0) -> Tuple[List[Optional[Optional[List[Tuple[Tuple[int, int], int]]]]], Set]:
    y_len = len(grid)
    x_len = len(grid[0])

    start_t = time.time()
    # note that resv_tbl and resv_locs are (y, x) - I'm pretty sure?
    # Convert node indexes from tuple to single int. -> Required for njit
    # yx -> ind     y * x_len + x
    resv_tbl_ind = set(map(lambda x: (x[0][0] * x_len + x[0][1], x[1]), resv_tbl))
    resv_locs_ind = set(map(lambda x: x[0] * x_len + x[0], resv_locs))
    # resv_locs_ind = {key[0] * x_len + key[1]: resv_locs[key] for key in resv_locs}

    print(time.time() - start_t)
    start_t = time.time()
    # resv_locs_vals_List = typed.List()
    # for val in resv_locs_vals:
    #     resv_locs_vals_List.append(typed.List(val))

    print(time.time() - start_t)

    cooperative_astar_path_njit(grid, sources[0], targets[0], resv_tbl_ind, resv_locs_ind, 0, 250)

    # test_njit_fn(resv_tbl_ind, resv_locs_ind)
    pass


def test_njit():
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)
    graph = GridGraph(grid, only_full_G=True).get_full_G()
    plot_graph(graph, "G.png")

    # ta = TaskAssigner(grid, set(), 1)
    # ta.inc_timestep_by_n(100)
    # tasks = [ta.get_ready_task() for _ in range(10)]

    # sources = [task.pickup_point for task in tasks]
    # targets = [task.dropoff_point for task in tasks]
    sources = [(0, 20), (0, 15)]
    targets = [(0, 15), (0, 25)]

    # (0, 26) (0, 15)
    # print(tasks)

    resv_tbl = set()

    # sources = [(0, 0), (5, 36)]
    # targets = [(5, 36), (0, 0)]

    paths, resv_tbl = cooperative_astar_path(graph, sources, targets, man_dist, resv_tbl=resv_tbl)
    # resv_locs = {(-1, -1): [(-1, -1), (-1, -1)]}
    # resv_locs = {(1, 1): [(0, 1), (2, 5)]}
    resv_locs = {(1, 1)}
    # resv_locs = typed.Dict()
    cooperative_astar_path_fast(np.array(grid), sources, targets, resv_tbl, resv_locs)

    # cooperative_astar_path_njit(sources, targets, resv_tbl, 0)
    pass


def main():
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)
    graph = GridGraph(grid, only_full_G=True).get_full_G()
    plot_graph(graph, "G.png")

    ta = TaskAssigner(grid)
    ta.increment_timestep_by_n(100)
    tasks = [ta.get_ready_task() for _ in range(10)]

    # sources = [task.pickup_point for task in tasks]
    # targets = [task.dropoff_point for task in tasks]
    sources = [(0, 20), (0, 15)]
    targets = [(0, 15), (0, 25)]

    # (0, 26) (0, 15)
    # print(tasks)

    resv_tbl = set()

    # sources = [(0, 0), (5, 36)]
    # targets = [(5, 36), (0, 0)]

    paths, resv_tbl = cooperative_astar_path(graph, sources, targets, man_dist, resv_tbl=resv_tbl)

    path_nodes = [[el[0] for el in path] for path in paths]

    vis = VisGrid(grid, (800, 400), 25, tick_time=0.2)
    vis.window.getMouse()
    vis.animate_multi_path(path_nodes, is_pos_xy=False)
    vis.window.getMouse()

    # print(paths[0])
    pass


def test_cython():
    import CooperativeAStarFast
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)
    grid = np.array(grid)
    # np_arr = np.ones((10, 10), dtype=int)
    # print(grid.shape)
    source = (0, 0)
    target = (10, 10)
    resv_tbl = {((1, 1), 1)}
    resv_locs = {(2, 2)}
    path = CooperativeAStarFast.cooperative_asar_path(grid, source[0], source[1],
                                               target[0], target[1], resv_tbl, resv_locs, 0)
    print(path)
    # pass


if __name__ == "__main__":
    # main()
    # test_njit()
    test_cython()
