import networkx as nx
from networkx.algorithms.shortest_paths.astar import cooperative_astar_path
from Benchmark import Warehouse
from MAPD.TaskAssigner import *
from GlobalObjs.GraphNX import get_strong_oriented_graph

if __name__ == "__main__":
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)
    G = get_strong_oriented_graph(grid)

    ta = TaskAssigner(grid, task_frequency=5)  # , is_printing=True)
    ta.increment_timestep_by_n(50)

    # print(f"No of ready tasks - {len(ta._ready_tasks)}")

    # ready_tasks = [task for task in ta.get_ready_tasks_gen()]
    resv_tbl = set()

    # print(ready_tasks)

    agents = generate_n_agents(5)
    #
    max_timesteps = 50
    #
    for t in range(max_timesteps):
        ready_tasks = []
        for agent in agents:
            if agent.is_ready():
                # give task
                pass
            else:
                pass
        # for ready_agent in get_all_ready_agents(agents):
        #     ready_task = ta.get_ready_task()
        #     if ready_task is None:
        #         continue
        #     else:
        #         paths, resv_tbl, max_t = cooperative_astar_path(G, [ready_task.pickup_point],
        #                                                         [ready_task.dropoff_point],
        #                                                         resv_tbl=resv_tbl, start_t=t)
        #         ready_agent.assign_task(ready_task, paths[0])
        #         # ready_tasks.append(ready_task)
        #
        # ta.increment_timestep()
        # inc_timestep_all_agents(agents)

#
# for each timestep
#
#   available_agents = get_available_agents()
#
