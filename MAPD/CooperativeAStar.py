import networkx as nx
from networkx.algorithms.shortest_paths.astar import astar_path
from Benchmark import Warehouse
from MAPD.TaskAssigner import *
from GlobalObjs.GraphNX import GridGraph
from GlobalObjs import GraphNX
from Visualisations.Vis import VisGrid
from MAPD import Agent
import os
import copy
from copy import deepcopy

from heapq import heappush, heappop
from itertools import count
import pickle
from typing import Optional


def _weight_function(G, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def cooperative_astar_path(G, sources, targets, heuristic=None, weight="weight", resv_tbl=None, start_t=0):
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
    paths = []
    if resv_tbl is None:
        resv_tbl = set()
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

    return paths, resv_tbl, max_t


class History:
    def __init__(self):
        self.orig_grid_graph = None
        self.t_history = []
        self.task_assigner_history = {}
        self.agents_history = {}
        self.ready_agents_history = {}
        self.ready_tasks_history = {}

        self.resv_tbl_history = {}  # t -> agent -> []
        self.grid_graph_history = {}
        self.paths_history = {}
        self.astar_arguments_history = {}
        self.astar_outputs_history = {}

    def add_agents(self, t, agents):
        agents_copy = [copy.deepcopy(agent) for agent in agents]
        self.agents_history[t] = agents_copy


def main():
    from timeit import default_timer as timer

    logging = 1

    is_plotting = False
    num_agents = 5
    is_printing = True
    is_saving_history = True
    max_t = 250

    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)

    first_locs = [(i*3, 0) for i in range(num_agents)]
    agents = Agent.generate_n_agents(num_agents, first_locs)
    # for i in range(num_agents):
    #     agents[i].first_loc = (i*3, 0)
    # agents[0].first_loc

    start_graph_t = timer()

    if is_printing:
        print("Creating graph...")
    grid_graph = GridGraph(grid)

    if is_plotting:
        if is_printing:
            print("Plotting graph...")

        # clear folder
        dir = os.getcwd() + '\\plots'
        for f in os.listdir(dir):
            if "png" in f:
                os.remove(os.path.join(dir, f))

        grid_graph.plot_full_G("plots/full_G.png")
        grid_graph.plot_G("plots/G.png")

    # G = grid_graph.get_G()
    unreachable_nodes = grid_graph.get_unreachable_nodes()

    end_graph_t = timer()

    grid_graph_dict = {}
    for agent in agents:
        grid_graph_dict[agent.id] = copy.deepcopy(grid_graph)

    hist = History()
    if is_saving_history:
        hist.orig_grid_graph = copy.deepcopy(grid_graph)

    start_mapd_t = timer()
    ta = TaskAssigner(grid, unreachable_nodes, task_frequency=2, is_printing=True)  # , is_printing=True)

    prev_task = None

    t = 0

    resv_tbl = set()

    if is_printing:
        print("Starting path finding...")
    while t < max_t:
        if logging > 1:
            print(f"t = {t}")

        ready_agents = get_all_ready_agents(agents)

        if is_saving_history:
            hist.add_agents(t, agents)
            hist.t_history.append(t)
            hist.task_assigner_history[t] = copy.deepcopy(ta)
            hist.grid_graph_history[t] = {}
            hist.resv_tbl_history[t] = {}
            hist.ready_tasks_history[t] = {}
            hist.astar_arguments_history[t] = {}
            hist.astar_outputs_history[t] = {}
            hist.paths_history[t] = {}

        for agent in ready_agents:
            if logging > 1:
                print(f"\tAgent {agent.id} ready")

            ready_task = ta.get_ready_task()
            if logging > 1:
                print(f"\tReady task: {ready_task}")

            if ready_task:
                if logging > 1:
                    print(f"\tTask {ready_task.id} assigned to agent {agent.id}")
                ready_task.timestep_assigned = t

                if is_saving_history:
                    hist.resv_tbl_history[t][agent.id] = []
                    hist.astar_arguments_history[t][agent.id] = []
                    hist.astar_outputs_history[t][agent.id] = []
                    hist.paths_history[t][agent.id] = []

                curr_grid_graph = grid_graph_dict[agent.id]

                ta.task_history.append({"task": ready_task, "agent_id": agent.id})

                # Remove relevant old access points
                if agent.prev_start_loc not in [None, agent.first_loc, ready_task.pickup_point,
                                                ready_task.dropoff_point, agent.loc]:
                    curr_grid_graph.remove_access_points(agent.prev_start_loc)
                if agent.prev_pickup_loc not in [None, ready_task.pickup_point, ready_task.dropoff_point]:
                    curr_grid_graph.remove_access_points(agent.prev_pickup_loc)

                # Add relevant new access points
                # Pickup Point
                if ready_task.pickup_point not in [agent.prev_start_loc, agent.prev_pickup_loc]:
                    curr_grid_graph.add_access_points(ready_task.pickup_point)
                # Dropoff Point
                if ready_task.dropoff_point not in [agent.prev_start_loc, agent.prev_pickup_loc, agent.loc]:
                    curr_grid_graph.add_access_points(ready_task.dropoff_point)

                if is_plotting:
                    curr_grid_graph.plot_G(f"G_{t}.png")
                G = curr_grid_graph.get_G()

                if is_saving_history:
                    hist.astar_arguments_history[t][agent.id].append(
                        deepcopy([G, agent.loc, ready_task.pickup_point, resv_tbl, t]))

                paths, resv_tbl, _ = cooperative_astar_path(G, [agent.loc],  [ready_task.pickup_point],
                                                            GraphNX.man_dist, 'weight', resv_tbl=resv_tbl, start_t=t)

                pickup_t = paths[0][-1][1]
                path_to_pickup = [tup[0] for tup in paths[0]]
                # path_to_pickup = paths[0]

                if is_saving_history:
                    hist.astar_arguments_history[t][agent.id].append(
                        deepcopy([G, ready_task.pickup_point, ready_task.dropoff_point, resv_tbl, pickup_t]))
                    hist.astar_outputs_history[t][agent.id].append(deepcopy([paths, resv_tbl]))

                paths, resv_tbl, _ = cooperative_astar_path(G, [ready_task.pickup_point], [ready_task.dropoff_point],
                                                            GraphNX.man_dist, 'weight', resv_tbl=resv_tbl, start_t=pickup_t)

                path_to_goal = [tup[0] for tup in paths[0]]
                # path_to_goal = paths[0]
                # path_to_pickup = astar_path(G, agent.loc, ready_task.pickup_point, GraphNX.man_dist, 'weight')
                # path_to_goal = astar_path(G, ready_task.pickup_point, ready_task.dropoff_point, GraphNX.man_dist, 'weight')
                path = path_to_pickup + path_to_goal[1:]
                agent.assign_task(ready_task, path)

                if is_saving_history:
                    hist.ready_tasks_history[t][agent.id] = deepcopy(ready_task)
                    hist.grid_graph_history[t][agent.id] = deepcopy(grid_graph)
                    hist.astar_outputs_history[t][agent.id].append(deepcopy([paths, resv_tbl]))

            else:
                break

        if is_saving_history:
            hist.ready_agents_history[t] = copy.deepcopy(ready_agents)
            hist.agents_history[t] = deepcopy(agents)

        t += 1
        ta.increment_timestep()
        Agent.inc_timestep_all_agents(agents)
        # agent.inc_timestep()

    if is_saving_history:
        with open("hist.pkl", "wb") as f:
            pickle.dump(hist, f)

    if is_printing:
        print("Path finding finished...")

    if is_plotting:
        if is_printing:
            print("Plotting graph...")

        grid_graph.plot_G(f"plots/G_final.png")

    end_mapd_t = timer()
    # end = timer()

    # print(f"#############")
    # print(f"#  AGENT {agent.id}  #")
    # print(f"#############\n")
    # print(f"TASK HISTORY:")
    # for i, task in enumerate(agent.task_history):
    #     print(f"Task {i} - {task}")
    # print(f"\nPATH HISTORY:")
    #
    # for i, path in enumerate(agent.path_history):
    #     print(f"Path {i} - {path}")
    #
    # full_path_sparse = [el for sublist in agent.path_history for el in sublist]
    #
    # full_path = []
    # for i in range(len(full_path_sparse) - 1):
    #     full_path.append(full_path_sparse[i])
    #     nodes_along_edge = GridGraph._nodes_along_edge((full_path_sparse[i], full_path_sparse[i+1]))
    #     if len(nodes_along_edge) > 0:
    #         full_path.extend(nodes_along_edge)
    #     # full_path.append(full_path_sparse[i+1])
    #
    # full_path.append(full_path_sparse[-1])
    # print(f"\nFULL PATH:\n{full_path}")
    full_paths = []

    for agent in agents:
        print(f"#############")
        print(f"#  AGENT {agent.id}  #")
        print(f"#############\n")
        print(f"TASK HISTORY:")
        for task in agent.task_history:
            print(f"Task {task.id} assigned @ {task.timestep_assigned} - {task}")
        print(f"\nPATH HISTORY:")

        for i, path in enumerate(agent.path_history):
            print(f"Path {i} (len = {len(path[1])}) - {path}")

        paths = [el[1] for el in agent.path_history]
        full_path_sparse = [el for sublist in paths for el in sublist]

        # full_path = []
        # for i in range(len(full_path_sparse) - 1):
        #     full_path.append(full_path_sparse[i])
        #     nodes_along_edge = GridGraph.nodes_along_edge((full_path_sparse[i], full_path_sparse[i+1]))
        #     if len(nodes_along_edge) > 0:
        #         full_path.extend(nodes_along_edge)
        #     # full_path.append(full_path_sparse[i+1])
        #
        # full_path.append(full_path_sparse[-1])

        full_path = agent.get_full_path()
        print(f"\nFULL PATH:\n{full_path}")
        full_paths.append(full_path)
        # full_paths_dict[agent.id] = full_path

    graph_t_elapsed = end_graph_t - start_graph_t
    mapd_t_elapsed = end_mapd_t - start_mapd_t

    print(f"\nTime Elapsed\n------------------")  # (s): {end-start}")
    print(f"Graph Creation(s): {graph_t_elapsed}")
    print(f"MAPD(s): {mapd_t_elapsed}")
    print(f"Total(s): {graph_t_elapsed+mapd_t_elapsed}")
    #

    max_len = max(len(arr) for arr in full_paths)
    for i, path in enumerate(full_paths):
        if len(path) < max_len:
            end_path = [path[-1]] * (max_len - len(path))
            full_paths[i].extend(end_path)

    print("SENSE CHECKING:")
    for pos_ind in range(max_len):
        for agent_ind_1 in range(len(full_paths)):
            for agent_ind_2 in range(agent_ind_1+1, len(full_paths)):
                if full_paths[agent_ind_1][pos_ind] == full_paths[agent_ind_2][pos_ind]:
                    print(f"Collision between {agent_ind_1} and {agent_ind_2} at t = {pos_ind}, "
                          f"pos = {full_paths[agent_ind_1][pos_ind]}")

    # lens = [len(arr) for arr in full_paths]

    new_vis = VisGrid(grid, (800, 400), 25, tick_time=0.2)
    new_vis.window.getMouse()
    new_vis.animate_multi_path(full_paths, is_pos_xy=False)
    # new_vis.animate_path(full_paths_dict[0], is_pos_xy=False)
    new_vis.window.getMouse()


    # ta.increment_timestep_by_n(50)
    #
    # # print(f"No of ready tasks - {len(ta._ready_tasks)}")
    #
    # # ready_tasks = [task for task in ta.get_ready_tasks_gen()]
    # resv_tbl = set()
    #
    # # print(ready_tasks)
    #
    # agents = generate_n_agents(5)
    # #
    # max_timesteps = 50
    # #
    # for t in range(max_timesteps):
    #     ready_tasks = []
    #     for agent in agents:
    #         if agent.is_ready():
    #             # give task
    #             pass
    #         else:
    #             pass
    #     for ready_agent in get_all_ready_agents(agents):
    #         ready_task = ta.get_ready_task()
    #         if ready_task is None:
    #             continue
    #         else:
    #             paths, resv_tbl, max_t = cooperative_astar_path(G, [ready_task.pickup_point],
    #                                                             [ready_task.dropoff_point],
    #                                                             resv_tbl=resv_tbl, start_t=t)
    #             ready_agent.assign_task(ready_task, paths[0])
    #             # ready_tasks.append(ready_task)
    #
    #     ta.increment_timestep()
    #     inc_timestep_all_agents(agents)


def visualise_paths(grid, agents):
    full_paths = []
    for agent in agents:
        full_path = agent.get_full_path()
        full_paths.append(full_path)

    max_len = max(len(arr) for arr in full_paths)
    for i, path in enumerate(full_paths):
        if len(path) < max_len:
            end_path = [path[-1]] * (max_len - len(path))
            full_paths[i].extend(end_path)

    print("SENSE CHECKING:")
    for pos_ind in range(max_len):
        for agent_ind_1 in range(len(full_paths)):
            for agent_ind_2 in range(agent_ind_1+1, len(full_paths)):
                if full_paths[agent_ind_1][pos_ind] == full_paths[agent_ind_2][pos_ind]:
                    print(f"Collision between {agent_ind_1} and {agent_ind_2} at t = {pos_ind}, "
                          f"pos = {full_paths[agent_ind_1][pos_ind]}")

    new_vis = VisGrid(grid, (800, 400), 25, tick_time=0.2)
    new_vis.window.getMouse()
    new_vis.animate_multi_path(full_paths, is_pos_xy=False)
    # new_vis.animate_path(full_paths_dict[0], is_pos_xy=False)
    new_vis.window.getMouse()


def pickle_debug():
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)

    hist: Optional[History] = None
    with open("hist.pkl", "rb") as f:
        hist = pickle.load(f)

    final_t = hist.t_history[-1]
    final_agents = hist.agents_history[final_t]

    visualise_paths(grid, final_agents)

    print(hist)


if __name__ == "__main__":
    main()
    # pickle_debug()
    # for i in range(50):
    #     pass

#
# for each timestep
#
#   available_agents = get_available_agents()
#
