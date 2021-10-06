import networkx as nx
from networkx.algorithms.shortest_paths.astar import cooperative_astar_path, astar_path
from Benchmark import Warehouse
from MAPD.TaskAssigner import *
from GlobalObjs.GraphNX import GridGraph
from GlobalObjs import GraphNX
from Visualisations.Vis import VisGrid


def main():
    from timeit import default_timer as timer

    is_plotting = True

    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)

    start_graph_t = timer()

    grid_graph = GridGraph(grid)

    if is_plotting:
        grid_graph.plot_full_G("full_G.png")
        grid_graph.plot_G("G.png")

    G = grid_graph.get_G()
    unreachable_nodes = grid_graph.get_unreachable_nodes()

    end_graph_t = timer()

    start_mapd_t = timer()
    ta = TaskAssigner(grid, unreachable_nodes, task_frequency=5)  # , is_printing=True)

    prev_task = None

    t = 0
    max_t = 50
    agent = Agent(0)
    while t < max_t:
        if agent.is_ready():
            ready_task = ta.get_ready_task()
            ta.task_history.append({"task": ready_task, "agent_id": agent.id})
            if ready_task:
                if prev_task is not None:
                    grid_graph.remove_access_points(prev_task.pickup_point)

                grid_graph.add_access_points(ready_task.pickup_point)
                grid_graph.add_access_points(ready_task.dropoff_point)
                if is_plotting:
                    grid_graph.plot_G(f"G_{t}.png")
                G = grid_graph.get_G()
                path_to_pickup = astar_path(G, agent.loc, ready_task.pickup_point, GraphNX.man_dist, 'weight')
                path_to_goal = astar_path(G, ready_task.pickup_point, ready_task.dropoff_point, GraphNX.man_dist, 'weight')
                path = path_to_pickup + path_to_goal[1:]
                agent.assign_task(ready_task, path)

                if prev_task is not None:
                    grid_graph.remove_access_points(prev_task.dropoff_point)

                prev_task = ready_task

        t += 1
        ta.increment_timestep()
        agent.inc_timestep()

    if is_plotting:
        grid_graph.plot_G(f"G_final.png")

    end_mapd_t = timer()
    # end = timer()

    print(f"TASK HISTORY:")
    for i, task in enumerate(agent.task_history):
        print(f"Task {i} - {task}")
    print(f"\nPATH HISTORY:")

    for i, path in enumerate(agent.path_history):
        print(f"Path {i} - {path}")

    full_path_sparse = [el for sublist in agent.path_history for el in sublist]

    full_path = []
    for i in range(len(full_path_sparse) - 1):
        full_path.append(full_path_sparse[i])
        nodes_along_edge = GridGraph._nodes_along_edge((full_path_sparse[i], full_path_sparse[i+1]))
        if len(nodes_along_edge) > 0:
            full_path.extend(nodes_along_edge)
        # full_path.append(full_path_sparse[i+1])

    full_path.append(full_path_sparse[-1])
    print(f"\nFULL PATH:\n{full_path}")

    graph_t_elapsed = end_graph_t - start_graph_t
    mapd_t_elapsed = end_mapd_t - start_mapd_t

    print(f"\nTime Elapsed\n------------------")  # (s): {end-start}")
    print(f"Graph Creation(s): {graph_t_elapsed}")
    print(f"MAPD(s): {mapd_t_elapsed}")
    print(f"Total(s): {graph_t_elapsed+mapd_t_elapsed}")

    new_vis = VisGrid(grid, (800, 400), 25, tick_time=0.2)
    new_vis.window.getMouse()
    new_vis.animate_path(full_path, is_pos_xy=False)
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


if __name__ == "__main__":
    main()

#
# for each timestep
#
#   available_agents = get_available_agents()
#
