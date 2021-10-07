# Based off [Ma et al. 2017]

from typing import Type, Tuple, List, Set, Optional, Dict

from GlobalObjs.GraphNX import GridGraph, plot_graph
from Benchmark import Warehouse
from Cooperative_AStar.CooperativeAStar import cooperative_astar_path, man_dist
from Visualisations.Vis import VisGrid

import numpy as np
from numpy import random


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

        pass

    def _process_input_grid(self):
        grid = self._grid
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                curr_el = grid[y][x]
                if curr_el not in self._unreachable_locs:
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
        assert self._curr_path[0] == self.curr_loc, "Path must start at agent's current location"

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

        if self._curr_path[0] == self.curr_loc:
            raise Exception("Path must start at agent's current location")
        # assert self._curr_path[0] == self.curr_loc,

        self._is_stationary = False
        self._reached_goal = False

    def inc_timestep(self):
        self._curr_t += 1
        self.move_along_path()

    def is_ready(self) -> bool:
        return self._is_stationary or self._reached_goal


class Token:
    def __init__(self, no_agents):
        self._no_agents = no_agents
        # each path is a space-time path, i.e. ((0, 0), 1) means agent is at (0, 0) at timestep = 1
        self._paths: List[List[Tuple[Tuple, int]]] = [[]]*no_agents
        self._is_stationary_list: List[bool] = [False]*no_agents

        # self.resv_tbl = set()
        # self.resv_locs = set()
        # self.path_end_locs = set()
        pass

    def update_path(self, agent_ind: int, path: List[Tuple[Tuple[int, int], int]], is_stationary):
        self._paths[agent_ind] = path
        self._is_stationary_list[agent_ind] = is_stationary

    def has_path_ending_in(self, locs: List[Tuple]) -> bool:
        for path in self._paths:
            if len(path) > 0 and path[-1][0] in locs:
                return True
        return False

    def get_resv_tbl(self) -> Set[Tuple[Tuple[int, int], int]]:
        resv_tbl: Set[Tuple[Tuple[int, int], int]] = set()
        for agent_ind in range(self._no_agents):
            if not self._is_stationary_list[agent_ind]:
                for el in self._paths[agent_ind]:
                    resv_tbl.add(el)

        return resv_tbl

    def get_resv_locs(self) -> Dict[Tuple[int, int], int]: # -> List[Tuple[Tuple[int, int], int]]:
        # resv_locs: List[Tuple[Tuple[int, int], int]] = []
        resv_locs: Dict[Tuple[int, int], int] = {}
        for agent_ind in range(self._no_agents):
            if self._is_stationary_list[agent_ind]:
                resv_locs[self._paths[agent_ind][-1][0]] = self._paths[agent_ind][-1][1]
        return resv_locs


class TokenPassing:
    def __init__(self, grid, no_agents, start_locs, non_task_endpoints, max_t, unreachable_locs=None, task_frequency=1, start_t=0):
        self._no_agents = no_agents
        self._grid = grid
        self._max_t = max_t
        self._non_task_endpoints: List[Tuple[int, int]] = non_task_endpoints

        grid_graph = GridGraph(grid, only_full_G=True)
        self._graph = grid_graph.get_full_G()

        self._agents = [Agent(i, start_locs[i]) for i in range(no_agents)]

        self._token = Token(no_agents)

        for agent_ind in range(no_agents):
            self._token.update_path(agent_ind, [(start_locs[agent_ind], start_t)], True)
        # for start_loc in start_locs:
        #     self._token.update_path()
        #     self._token.resv_locs.add(start_loc)
        if unreachable_locs is None:
            unreachable_locs = set()

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

                    resv_tbl = token.get_resv_tbl()
                    resv_locs = token.get_resv_locs()

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
                    token.update_path(agent.id, path, False)
                else:
                    is_stationary_valid = True  # if no task has goal at agent's current location
                    for task in tasks:
                        if task.dropoff_point == agent.curr_loc:
                            is_stationary_valid = False
                            break

                    if is_stationary_valid:
                        # Update agent's path in token with stationary path
                        token.update_path(agent.id, [(agent.curr_loc, curr_t)], True)

                    else:
                        goal = None
                        # Update agent's path in token with deadlock avoidance path
                        # i.e. path to non-occupied non-task endpoint
                        for endpoint in self._non_task_endpoints:
                            if not token.has_path_ending_in([endpoint]):
                                goal = endpoint
                                break
                        if goal is None:
                            raise Exception("No valid non-task endpoints found :(")

                        # find path to goal and update agent & token
                        source = agent.curr_loc
                        resv_tbl = token.get_resv_tbl()
                        resv_locs = token.get_resv_locs()

                        paths, _ = cooperative_astar_path(graph, [source], [goal], resv_tbl=resv_tbl, resv_locs=resv_locs, start_t=curr_t)
                        path = paths[0]
                        agent.assign_avoidance_path(path)
                        token.update_path(agent.id, path, False)

                agent.inc_timestep()

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
    grid = Warehouse.txt_to_grid("map_warehouse_1.txt", use_curr_workspace=True, simple_layout=False)
    y_len = len(grid)
    x_len = len(grid[0])

    non_task_endpoints = [(y, 0) for y in range(y_len)]
    no_agents = 5
    start_locs = non_task_endpoints[:no_agents]

    tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, 250, task_frequency=5)
    final_agents = tp.compute()

    for agent in final_agents:
        print(f"############")
        print(f"# AGENT {agent.id:2} #")
        print(f"############")
        print(f"Path History: {agent.path_history}")
        print(f"Task History: {agent.task_history}")
        print(f"Current Task: {agent._curr_task}")
        print(f"Current Path: {agent._curr_path}")
    # visualise_paths(grid, final_agents)
    # plot_graph(tp._graph, "tp_G.png")


if __name__ == "__main__":
    main()
