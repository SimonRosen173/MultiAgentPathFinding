from typing import List, Tuple
import numpy as np
from GlobalObjs.GraphNX import GridGraph, man_dist

__all__ = ['generate_n_agents', 'get_ready_agent', 'Agent', 'inc_timestep_all_agents', 'get_all_ready_agents']


class Agent:
    def __init__(self, id, start_timestep=0, first_loc=(0, 0)):
        self.id = id

        self.start_timestep = start_timestep
        self.curr_timestep = start_timestep

        self.task_start_timestep = -1
        self.task_end_timestep = np.inf

        self._path_to_pickup = []  # Path from curr location to pickup point
        self._path_to_dropoff = []  # Path from pickup point to drop off point

        self.curr_path = []

        self.path_history = []
        self.task_history = []

        self.prev_start_loc = None
        self.prev_pickup_loc = None

        self.curr_task = None
        self.prev_task = None
        self.tasks_complete = []
        self._is_task_complete = True

        self.loc_in_path = -1  # Index of node agent has most recently visited or is currently on
        self.loc = first_loc
        self.first_loc = first_loc

    def inc_timestep(self, n=1):
        self.curr_timestep += 1
        if self.curr_timestep >= self.task_end_timestep:
            self.tasks_complete.append(self.curr_task)
            self.loc = self.curr_path[-1]
            self.curr_task = None
            self._is_task_complete = True

    def assign_task(self, task, path):
        self.curr_task = task
        self._is_task_complete = False

        self.curr_path = path

        # tmp =
        time_of_path = sum([man_dist(path[i], path[i+1]) for i in range(len(path)-1)])
        time_of_path += man_dist(path[-2], path[-1])

        self.task_start_timestep = self.curr_timestep
        # time_of_path = sum([el[1] for el in weighted_path])
        self.task_end_timestep = self.task_start_timestep + time_of_path



        # Check that this is correct
        self.path_history.append((self.curr_timestep, path))
        self.task_history.append(task)
        # self.tasks_complete.append(task)
        # self.paths.append(path)

    def is_ready(self):
        return self._is_task_complete
        # return not self.curr_task or self.curr_timestep >= self.task_end_timestep

    # TODO
    def get_full_path(self):
        # paths = []
        full_path = []
        curr_path = []
        t = 0

        def sparse_to_dense_path(sparse_path):
            dense_path = []
            for i in range(len(sparse_path) - 1):
                dense_path.append(sparse_path[i])
                nodes_along_edge = GridGraph.nodes_along_edge((sparse_path[i], sparse_path[i+1]))
                if len(nodes_along_edge) > 0:
                    dense_path.extend(nodes_along_edge)
            dense_path.append(sparse_path[-1])
            return dense_path

        # first path
        first_t = self.path_history[0][0]
        if first_t != t:
            full_path = [self.first_loc]*(first_t-t)
        curr_path = sparse_to_dense_path(self.path_history[0][1])
        t += len(curr_path)
        full_path.extend(curr_path)

        for i in range(1, len(self.path_history)):
            curr_t, curr_path = self.path_history[i]
            if curr_t > t:
                tmp = [full_path[-1]]*(curr_t-t)
                full_path.extend(tmp)
                t = curr_t

            curr_dense = sparse_to_dense_path(self.path_history[i][1])
            t += len(curr_dense)
            full_path.extend(curr_dense)

            # pass
        # for t, path in self.path_history:
        #     pass
        return full_path

    # def is_task_complete(self):
    #     # i.e. has task and task has completed
    #     return self._is_task_complete
    #     # return self.curr_task and self.task_end_timestep > self.curr_timestep


def generate_n_agents(n, first_locs: List[Tuple], start_id=0, start_timestep=0):
    agents = []
    for i in range(n):
        agents.append(Agent(i + start_id, start_timestep, first_loc=first_locs[i]))
    return agents


def inc_timestep_all_agents(agents: List[Agent], n=1):
    for agent in agents:
        agent.inc_timestep(n)


def get_ready_agent(agents: List[Agent]):
    for agent in agents:
        if agent.is_ready():
            return agent
    return None


def get_all_ready_agents(agents: List[Agent]):
    return list(filter(lambda agent: True if agent.is_ready() else False, agents))

#