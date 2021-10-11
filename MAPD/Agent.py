from typing import List, Tuple


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