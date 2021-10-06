from typing import List

__all__ = ['generate_n_agents', 'get_ready_agent', 'Agent', 'inc_timestep_all_agents', 'get_all_ready_agents']


class Agent:
    def __init__(self, id, start_timestep=0, start_loc=(0, 0)):
        self.id = id

        self.start_timestep = start_timestep
        self.curr_timestep = start_timestep

        self.task_start_timestep = -1
        self.task_end_timestep = -1

        self._path_to_pickup = []  # Path from curr location to pickup point
        self._path_to_dropoff = []  # Path from pickup point to drop off point

        self.curr_path = []

        self.path_history = []
        self.task_history = []

        self.curr_task = None
        self.tasks_complete = []

        self.loc_in_path = -1  # Index of node agent has most recently visited or is currently on
        self.loc = start_loc

    def inc_timestep(self, n=1):
        self.curr_timestep += 1
        if self.curr_task and self.task_end_timestep > self.curr_timestep:
            self.tasks_complete.append(self.curr_task)
            self.loc = self.curr_path[-1]
            self.curr_task = None

    def assign_task(self, task, path):
        self.curr_task = task
        self.curr_path = path

        self.task_start_timestep = self.curr_timestep
        time_of_path = sum([el[1] for el in path])
        self.task_end_timestep = self.task_start_timestep + time_of_path

        self.path_history.append(path)
        self.task_history.append(task)
        # self.tasks_complete.append(task)
        # self.paths.append(path)

    def is_ready(self):
        return not self.curr_task or self.task_end_timestep > self.curr_timestep

    def is_task_complete(self):
        # i.e. has task and task has completed
        return self.curr_task and self.task_end_timestep > self.curr_timestep


def generate_n_agents(n, start_id=0, start_timestep=0):
    agents = []
    for i in range(n):
        agents.append(Agent(i + start_id, start_timestep))
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