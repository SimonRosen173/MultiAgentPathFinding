from heapq import heappush, heappop
from itertools import count
import networkx as nx
from Benchmark import Warehouse
from GlobalObjs.GraphNX import GridGraph, plot_graph
from Visualisations.Vis import VisGrid
from MAPD.TaskAssigner import TaskAssigner
from typing import Dict, List, Tuple, Set, Optional


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


def cooperative_astar_path(G, sources, targets, heuristic=man_dist, weight="weight",
                           resv_tbl: Optional[Set[Tuple[Tuple[int, int], int]]] = None,
                           # resv_locs - locations reserved from some timestep onwards
                           resv_locs: Optional[Dict[Tuple[int, int], int]] = None,
                           start_t=0) -> Tuple[List[Optional[Optional[List[Tuple[Tuple[int, int], int]]]]], Set]:
    assert len(sources) == len(targets), "Length of sources must equal length of targets"

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

                curr_t = curnode[1] + curr_weight
                ncost = dist + curr_weight # weight(curnode[0], neighbor, w)
                # curr_t = ncost

                if (neighbor, curr_t) not in resv_tbl:
                    # I.e. if neighbor is a stationary agent
                    if neighbor in resv_locs and resv_locs[neighbor] >= curr_t:
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
            paths.append(None)
            # raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} for agent {agent_ind}")

    return paths, resv_tbl


def main():
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)
    graph = GridGraph(grid, only_full_G=True).get_full_G()
    plot_graph(graph, "G.png")

    ta = TaskAssigner(grid)
    ta.increment_timestep_by_n(100)
    tasks = [ta.get_ready_task() for _ in range(10)]

    sources = [task.pickup_point for task in tasks]
    targets = [task.dropoff_point for task in tasks]
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


if __name__ == "__main__":
    main()
