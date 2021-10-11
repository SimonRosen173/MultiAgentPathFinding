# Based off [Ma et al. 2017]

from typing import Type, Tuple, List, Set, Optional, Dict

from GlobalObjs.GraphNX import GridGraph, plot_graph
from Benchmark import Warehouse
from Cooperative_AStar.CooperativeAStar import cooperative_astar_path, man_dist
from Visualisations.Vis import VisGrid

import numpy as np
from numpy import random
# random.seed(42)


class Task:
    def __init__(self, id, pickup_point, dropoff_point, timestep_created):
        self.id = id
        self.pickup_point = pickup_point
        self.dropoff_point = dropoff_point
        self.timestep_created = timestep_created
        self.timestep_assigned = -1
        self.timestep_completed = -1

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"TASK(id={self.id}) - pickup_point={self.pickup_point}, dropoff_point={self.dropoff_point}"


class TaskAssigner:
    EMPTY = 0
    PICKUP = 1
    DROPOFF = 2
    OBSTACLE = 3

    def __init__(self, grid, unreachable_locs, task_frequency):
        self._grid = grid
        self._unreachable_locs = unreachable_locs
        self._task_frequency = task_frequency

        self._pickup_points = []
        self._dropoff_points = []

        self._ready_tasks = []
        self._complete_tasks = []

        self._task_id_no = 0

        self._process_input_grid()

        self._curr_t = -1
        self.inc_timestep()
        # print(f"Random seed {}")

        pass

    def _process_input_grid(self):
        grid = self._grid
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                curr_el = grid[y][x]
                if (y, x) not in self._unreachable_locs:
                    if curr_el == TaskAssigner.PICKUP:
                        self._pickup_points.append((y, x))
                    elif curr_el == TaskAssigner.DROPOFF:
                        self._dropoff_points.append((y, x))
                    elif curr_el == TaskAssigner.OBSTACLE:
                        pass

    def _create_task(self):
        if len(self._pickup_points) - 1 < 0 or len(self._dropoff_points) - 1 < 0:
            print("Available pickup points and dropoff points depleted")
            return
        if len(self._pickup_points) - 1 == 0:
            pickup_point_ind = 0
        else:
            pickup_point_ind = random.randint(0, len(self._pickup_points) - 1)

        if len(self._dropoff_points) - 1 == 0:
            dropoff_point_ind = 0
        else:
            dropoff_point_ind = random.randint(0, len(self._dropoff_points) - 1)

        curr_t = self._curr_t
        task_id = self._task_id_no
        self._task_id_no += 1

        pickup_point = self._pickup_points[pickup_point_ind]
        dropoff_point = self._dropoff_points[dropoff_point_ind]

        # del self._pickup_points[pickup_point_ind]
        # del self._dropoff_points[dropoff_point_ind]

        new_task = Task(task_id, pickup_point, dropoff_point, curr_t)

        # print(f"Task Created = {str(new_task)}")

        self._ready_tasks.append(new_task)

    def get_ready_tasks(self):
        return self._ready_tasks

    def get_ready_task(self):
        if len(self._ready_tasks) > 0:
            return self._ready_tasks[0]
        else:
            return None

    def remove_task_from_ready(self, task: Task):
        for i in range(len(self._ready_tasks)):
            if self._ready_tasks[i] == task:
                del self._ready_tasks[i]
                break

    def task_complete(self, task: Task):
        # task.timestep_completed = self._curr_t
        # self._dropoff_points.append(task.dropoff_point)
        # self._pickup_points.append(task.pickup_point)
        self._complete_tasks.append(task)

    def inc_timestep(self):
        self._curr_t += 1
        if self._curr_t % self._task_frequency == 0:
            self._create_task()


class Agent:
    def __init__(self, id: int, start_loc: Tuple[int, int]):
        self.id = id
        self._curr_t = 0
        self.curr_loc = start_loc
        self._curr_path_loc = 0

        self._is_avoidance_path = False  # I.e. path to non-task endpoint

        self._curr_task = None
        self._curr_path = None
        self._path_to_pickup = None
        self._path_to_dropoff = None
        self._timestep_path_started = -1

        self._is_stationary = True  # I.e. following 'trivial path'

        self._reached_goal = False

        self.path_history = {}
        self.task_history = {}

    def move_along_path(self):
        if self._is_stationary or self._reached_goal:
            pass
        else:
            self._curr_path_loc += 1
            self.curr_loc = self._curr_path[self._curr_path_loc][0]

            if self._curr_path_loc == len(self._curr_path) - 1:  # I.e. if at last point in path
                self._reached_goal = True
                self._is_stationary = True

                if self._is_avoidance_path:
                    self.task_history[self._curr_t] = None
                    self.path_history[self._curr_t] = (self._timestep_path_started, None, None, self._curr_path)
                else:
                    self.task_history[self._curr_t] = self._curr_task
                    self.path_history[self._curr_t] = (self._timestep_path_started, self._path_to_pickup, self._path_to_dropoff, None)

                self._curr_task = None
                self._timestep_path_started = -1

    def assign_avoidance_path(self, path):
        self._path_to_pickup = None
        self._path_to_dropoff = None
        self._curr_task = None

        self._curr_path = path
        self._timestep_path_started = self._curr_t
        self._is_avoidance_path = True

        self._curr_path_loc = 0
        if self._curr_path[0][0] != self.curr_loc:
            raise Exception("Path must start at agent's current location")
        # assert self._curr_path[0][0] == self.curr_loc,

        self._is_stationary = False
        self._reached_goal = False

    def assign_task(self, task, path_to_pickup, path_to_dropoff):
        self._path_to_pickup = path_to_pickup
        self._path_to_dropoff = path_to_dropoff
        self._timestep_path_started = self._curr_t
        self._curr_task = task

        self._is_avoidance_path = False

        self._curr_path = path_to_pickup + path_to_dropoff[1:]
        self._curr_path_loc = 0

        if self._curr_path[0][0] != self.curr_loc:
            raise Exception("Path must start at agent's current location")
        # assert self._curr_path[0] == self.curr_loc,

        self._is_stationary = False
        self._reached_goal = False

    def inc_timestep(self):
        self._curr_t += 1
        self.move_along_path()

    def is_ready(self) -> bool:
        return self._is_stationary or self._reached_goal

    def get_full_path(self):
        path_history = self.path_history
        all_t = list(sorted(path_history.keys()))
        full_path = []
        for t in all_t:
            curr_path_hist = path_history[t]
            curr_path = []
            # (self._timestep_path_started, self._path_to_pickup, self._path_to_dropoff, None)
            _, path_to_pickup, path_to_dropoff, avoidance_path = curr_path_hist
            if avoidance_path is None:
                curr_path = path_to_pickup + path_to_dropoff[1:]
            else:
                curr_path = avoidance_path
            full_path.extend(curr_path)
        return full_path


# NOTE: Token must contain full paths of agents not just current
class Token:
    def __init__(self, no_agents, start_locs: List[Tuple[int, int]], non_task_endpoints: List[Tuple[int, int]],
                 start_t=0):
        self._resv_rbl = set()
        self._no_agents = no_agents
        self._non_task_endpoints = non_task_endpoints

        # each path is a space-time path, i.e. ((0, 0), 1) means agent is at (0, 0) at timestep = 1
        self._paths: List[List[Tuple[Tuple, int]]] = [[] for _ in range(no_agents)]
        # Pos and time interval agents are stationary
        self._stationary_list: List[List[Tuple[Tuple, int, int]]] = [[] for _ in range(no_agents)]
        self._is_stationary_list: List[bool] = [False]*no_agents

        for agent_ind in range(no_agents):
            self._paths[agent_ind] = [(start_locs[agent_ind], start_t)]
            self.add_stationary(agent_ind, start_locs[agent_ind], start_t)

        # self.resv_tbl = set()
        # self.resv_locs = set()
        # self.path_end_locs = set()
        pass

    def add_stationary(self, agent_id: int, pos: Tuple[int, int], start_t: int):
        if not self._is_stationary_list[agent_id]:
            # add_stationary sets agent specified as stationary
            self._is_stationary_list[agent_id] = True

            self._stationary_list[agent_id].append((pos, start_t, np.inf))

    # pos only required for error checking
    def _update_last_end_t(self, agent_id: int, pos: tuple[int, int], end_t: int):
        last_stationary = self._stationary_list[agent_id][-1]
        assert last_stationary[0] == pos, f"{last_stationary[0]} != {pos} - pos specified must correspond to last element in _stationary_list[agent_id]"
        last_stationary = (last_stationary[0], last_stationary[1], end_t)
        self._stationary_list[agent_id][-1] = last_stationary

    def add_to_path(self, agent_id: int, path: List[Tuple[Tuple[int, int], int]]):
        # If agent was stationary set end time of relevant element of _stationary_list
        if self._is_stationary_list[agent_id]:  # and len(self._paths[agent_id]) > 0 and len(self._stationary_list[agent_id]) > 0
            end_t = self._paths[agent_id][-1][1]
            last_stationary = self._stationary_list[agent_id][-1]
            last_stationary = (last_stationary[0], last_stationary[1], end_t)
            self._stationary_list[agent_id][-1] = last_stationary

        # add_to_path sets agent specified as not stationary
        self._is_stationary_list[agent_id] = False

        self._paths[agent_id].extend(path)

    def is_stationary(self, agent_id: int) -> bool:
        return self._is_stationary_list[agent_id]

    # def update_path(self, agent_ind: int, path: List[Tuple[Tuple[int, int], int]], is_stationary):
    #     self._paths[agent_ind] = path
    #     self._is_stationary_list[agent_ind] = is_stationary

    def has_path_ending_in(self, locs: List[Tuple]) -> bool:
        for path in self._paths:
            if len(path) > 0 and path[-1][0] in locs:
                return True
        for stat in self._stationary_list:
            last_stationary = stat[-1]
            if last_stationary[0] in locs:
                return True

        return False

    # Is there a non-task endpoint at pos excluding the non-task endpoint of specified agent
    def non_task_endpoint_at(self, pos: Tuple[int, int], agent_id):
        for curr_agent_id in range(self._no_agents):
            if curr_agent_id != agent_id and self._non_task_endpoints[curr_agent_id] == pos:
                return True
        return False

    # Reservation table for agent given should not include the prev path of that agent
    def get_resv_tbl(self, agent_id: int, curr_t: int) -> Set[Tuple[Tuple[int, int], int]]:
        resv_tbl: Set[Tuple[Tuple[int, int], int]] = set()
        # resv_tbl = self._resv_rbl
        for curr_agent_id in range(self._no_agents):
            if curr_agent_id != agent_id and not self._is_stationary_list[curr_agent_id]:
                for el in self._paths[curr_agent_id]:
                    if el[1] >= curr_t:  # NEW
                        resv_tbl.add(el)
                        # Reserve at next time step to avoid head-on/pass-through collisions
                        resv_tbl.add((el[0], el[1]+1))

        return resv_tbl

    def get_resv_locs(self, agent_id: int, curr_t: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        resv_locs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for curr_agent_id in range(self._no_agents):
            if curr_agent_id != agent_id:
                for stationary_loc in self._stationary_list[curr_agent_id]:
                    if stationary_loc[2] >= curr_t:  # NEW
                        if stationary_loc[0] not in resv_locs:
                            resv_locs[stationary_loc[0]] = [(stationary_loc[1], stationary_loc[2])]
                        else:
                            resv_locs[stationary_loc[0]].append((stationary_loc[1], stationary_loc[2]))
                resv_locs[self._non_task_endpoints[curr_agent_id]] = [(0, np.inf)]

        return resv_locs

    def get_last_path_locs(self):
        return [path[-1] for path in self._paths]

    def get_agents_with(self, loc):
        ids = []
        for agent_id in range(self._no_agents):
            if loc in self._paths[agent_id]:
                ids.append(agent_id)
        return ids

    # # Reserved locations for agent given should not include the prev path of that agent
    # def get_resv_locs(self, agent_id: int) -> Dict[Tuple[int, int], int]: # -> List[Tuple[Tuple[int, int], int]]:
    #     # resv_locs: List[Tuple[Tuple[int, int], int]] = []
    #     resv_locs: Dict[Tuple[int, int], int] = {}
    #     for curr_agent_id in range(self._no_agents):
    #         if curr_agent_id != agent_id and self._is_stationary_list[curr_agent_id]:
    #             resv_locs[self._paths[curr_agent_id][-1][0]] = self._paths[curr_agent_id][-1][1]
    #     return resv_locs


class TokenPassing:
    def __init__(self, grid, no_agents, start_locs, non_task_endpoints, max_t, unreachable_locs=None,
                 task_frequency=1, start_t=0, is_logging_collisions=False):
        self._no_agents = no_agents
        self._grid = grid
        self._max_t = max_t
        self._non_task_endpoints: List[Tuple[int, int]] = non_task_endpoints
        self._is_logging_collisions = is_logging_collisions

        grid_graph = GridGraph(grid, only_full_G=True)
        grid_graph.remove_non_reachable()
        self._graph = grid_graph.get_full_G()

        self._agents = [Agent(i, start_locs[i]) for i in range(no_agents)]

        self._token = Token(no_agents, start_locs, non_task_endpoints)

        # self._token.update_path(agent_ind, [(start_locs[agent_ind], start_t)], True)
        # for start_loc in start_locs:
        #     self._token.update_path()
        #     self._token.resv_locs.add(start_loc)
        unreachable_locs = set(grid_graph.get_unreachable_nodes())
        # if unreachable_locs is None:
        #     pass

        self._ta = TaskAssigner(grid, unreachable_locs, task_frequency)

    def compute(self):
        token = self._token
        agents = self._agents
        graph = self._graph
        ta = self._ta
        max_t = self._max_t

        curr_t = 0

        while curr_t < max_t:
            for agent in agents:
                # Skip agent if not ready
                if not agent.is_ready():
                    agent.inc_timestep()
                    continue

                # Do task stuff
                tasks = ta.get_ready_tasks()
                # I.e. task set prime in pseudo code
                tasks_prime = []
                for task in tasks:
                    # if not (task.pickup_point in token.path_end_locs or task.dropoff_point in token.path_end_locs):
                    if not token.has_path_ending_in([task.pickup_point, task.dropoff_point]):
                        tasks_prime.append(task)

                if len(tasks_prime) > 0:
                    # get task with pickup point closest to curr loc of agent
                    min_dist = np.inf
                    min_task: Optional[Task] = None
                    for task in tasks_prime:
                        curr_dist = man_dist(agent.curr_loc, task.pickup_point)
                        if curr_dist < min_dist:
                            min_task = task

                    resv_tbl = token.get_resv_tbl(agent.id, curr_t)
                    resv_locs = token.get_resv_locs(agent.id, curr_t)

                    ta.remove_task_from_ready(min_task)

                    paths, _ = cooperative_astar_path(graph, [agent.curr_loc], [min_task.pickup_point], resv_tbl=resv_tbl,
                                                      resv_locs=resv_locs, start_t=curr_t)
                    path_to_pickup = paths[0]
                    pickup_t = path_to_pickup[-1][1]

                    paths, _ = cooperative_astar_path(graph, [min_task.pickup_point], [min_task.dropoff_point], resv_tbl=resv_tbl,
                                                      resv_locs=resv_locs, start_t=pickup_t)
                    path_to_dropoff = paths[0]

                    agent.assign_task(min_task, path_to_pickup, path_to_dropoff)

                    path = path_to_pickup + path_to_dropoff[1:]

                    token.add_to_path(agent.id, path)
                    # token.update_path(agent.id, path, False)
                else:
                    is_stationary_valid = True  # if no task has goal at agent's current location
                    for task in tasks:
                        if task.dropoff_point == agent.curr_loc:
                            is_stationary_valid = False
                            break

                    if is_stationary_valid:
                        # Update agent's path in token with stationary path
                        token.add_stationary(agent.id, agent.curr_loc, curr_t)
                        # token.update_path(agent.id, [(agent.curr_loc, curr_t)], True)

                    else:
                        goal = None
                        # Update agent's path in token with deadlock avoidance path
                        # i.e. path to non-occupied non-task endpoint
                        # for endpoint in self._non_task_endpoints:
                        #     if not token.has_path_ending_in([endpoint]):
                        #         goal = endpoint
                        #         break
                        # if goal is None:
                        #     raise Exception("No valid non-task endpoints found :(")
                        goal = self._non_task_endpoints[agent.id]  # Each agent has unique non-task endpoint

                        # find path to goal and update agent & token
                        source = agent.curr_loc
                        resv_tbl = token.get_resv_tbl(agent.id, curr_t)
                        resv_locs = token.get_resv_locs(agent.id, curr_t)

                        # if agent.id == 7 and curr_t == 265:
                        #     last_locs = token.get_last_path_locs()
                        #     tmp = ((11, 2), 265) in resv_tbl
                        #     tmp_1 = ((11, 2), 264) in resv_tbl
                        #     tmp_3 = ((11, 2), 266) in resv_tbl
                        #     agents_ids = token.get_agents_with(((11, 2), 265))

                        paths, _ = cooperative_astar_path(graph, [source], [goal], resv_tbl=resv_tbl, resv_locs=resv_locs, start_t=curr_t)
                        path = paths[0]
                        agent.assign_avoidance_path(path)
                        # token.update_path(agent.id, path, False)
                        token.add_to_path(agent.id, path)

                agent.inc_timestep()

            if self._is_logging_collisions:
                for agent_id_1 in range(self._no_agents):
                    for agent_id_2 in range(agent_id_1 + 1, self._no_agents):
                        if self._agents[agent_id_1].curr_loc == self._agents[agent_id_2].curr_loc:
                            collide_agent_1 = self._agents[agent_id_1]
                            collide_agent_2 = self._agents[agent_id_2]
                            print(f"COLLISION(t = {curr_t}) - collision between agent {agent_id_1} and agent {agent_id_2} at "
                                  f"pos = {self._agents[agent_id_1].curr_loc}")
                pass

            curr_t += 1
            ta.inc_timestep()

        return agents


def visualise_paths(grid, agents: List[Agent]):
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


def main():
    grid = Warehouse.get_uniform_random_grid((22, 44), 560)

    # grid = Warehouse.txt_to_grid("map_warehouse_1.txt", use_curr_workspace=True, simple_layout=False)
    y_len = len(grid)
    x_len = len(grid[0])

    non_task_endpoints = [(y, 0) for y in range(y_len)]
    no_agents = 5
    start_locs = non_task_endpoints[:no_agents]

    tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, 500, task_frequency=1,
                      is_logging_collisions=True)
    final_agents = tp.compute()

    # new_vis.animate_multi_path(full_paths, is_pos_xy=False)
    # new_vis.animate_path(full_paths_dict[0], is_pos_xy=False)
    # new_vis.window.getMouse()

    for agent in final_agents:
        print(f"############")
        print(f"############")
        print(f"# AGENT {agent.id:2} #")
        print(f"############")
        print(f"Path History: {agent.path_history}")
        print(f"Task History: {agent.task_history}")
        print(f"Current Task: {agent._curr_task}")
        print(f"Current Path: {agent._curr_path}")
    # visualise_paths(grid, final_agents)
    # plot_graph(tp._graph, "tp_G.png")

    vis = VisGrid(grid, (800, 400), 25, tick_time=0.2)
    vis.window.getMouse()
    vis.animate_mapd(final_agents, is_pos_xy=False)
    vis.window.getMouse()


if __name__ == "__main__":
    main()
