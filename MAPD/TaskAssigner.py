from typing import List, Dict
import random
from Benchmark import Warehouse
from MAPD.Agent import *


class Task:
    def __init__(self, id, pickup_point, dropoff_point):
        self.pickup_point = pickup_point
        self.dropoff_point = dropoff_point
        self.is_at_dropoff = False
        self.time_at_dropoff = 0  # How long task has been at dropoff point
        self.id = id

    def __str__(self):
        return f"Task(id={self.id}, pickup_point={self.pickup_point}, droppoff_point={self.dropoff_point})"

    def __repr__(self):
        return self.__str__()


class TaskAssigner:
    EMPTY = 0
    PICKUP = 1
    DROPOFF = 2
    OBSTACLE = 3

    def __init__(self, input_grid, task_frequency=1, max_time_at_droppoff=2, is_printing=False):
        self._pickup_points = []
        self._dropoff_points = []
        self._input_grid = input_grid
        self._grid = [[False]*len(input_grid[0]) for _ in range(len(input_grid))]

        self._tasks_at_dropoff: List[Task] = []
        self._ready_tasks: List[Task] = []
        self._completed_tasks: List[Task] = []
        self._in_progress_tasks: Dict[Task] = {}

        self._process_input_grid()

        self._task_frequency = task_frequency
        self._curr_timestep = 0

        self._is_printing = is_printing

        self._max_time_at_droppoff = max_time_at_droppoff  # How long task stays at dropoff point

        self._no_tasks = 0

    def _process_input_grid(self):
        input_grid = self._input_grid
        grid = self._grid

        for y in range(len(input_grid)):
            for x in range(len(input_grid[0])):
                curr_el = input_grid[y][x]
                if curr_el == TaskAssigner.PICKUP:
                    self._pickup_points.append((x,y))
                    grid[y][x] = True
                elif curr_el == TaskAssigner.DROPOFF:
                    self._dropoff_points.append((x,y))
                    grid[y][x] = True
                elif curr_el == TaskAssigner.OBSTACLE:
                    grid[y][x] = True

        self._grid = grid

    def _generate_task(self):
        pickup_point_ind = random.randint(0, len(self._pickup_points) - 1)
        dropoff_point_ind = random.randint(0, len(self._dropoff_points) - 1)

        pickup_point = self._pickup_points[pickup_point_ind]
        dropoff_point = self._dropoff_points[dropoff_point_ind]

        del self._pickup_points[pickup_point_ind]
        # self.pickup_points.remove(pickup_point_ind)  # Removed until package put back

        task_id = self._no_tasks
        self._no_tasks += 1

        new_task = Task(task_id, pickup_point, dropoff_point)

        if self._is_printing:
            print(f"New task created - ID = {new_task.id}, pickup_point = {new_task.pickup_point}, "
              f"dropoff_point = {new_task.dropoff_point}")

        self._ready_tasks.append(new_task)

    def task_complete(self, task: Task):
        del self._in_progress_tasks[task.id]
        self._completed_tasks.append(task)
        self._pickup_points.append(task.pickup_point)  # Package put back

    def task_at_dropoff(self, task):
        del self._in_progress_tasks[task.id]
        self._tasks_at_dropoff.append(task)

    def get_ready_task(self):
        if len(self._ready_tasks) > 0:
            task = self._ready_tasks.pop(0)
            self._in_progress_tasks[task.id] = task
            return task
        else:
            return None

    def get_ready_tasks_gen(self):
        ready_task = self.get_ready_task()
        while ready_task:
            yield ready_task
            ready_task = self.get_ready_task()

    def increment_timestep_by_n(self, n):
        for _ in range(n):
            self.increment_timestep()

    def increment_timestep(self):
        if self._task_frequency > 0:
            if self._curr_timestep % self._task_frequency == 0:
                self._generate_task()
            self._curr_timestep += 1

        for i, task in enumerate(self._tasks_at_dropoff):
            task.time_at_dropoff += 1
            if task.time_at_dropoff >= self._max_time_at_droppoff:
                self._tasks_at_dropoff.remove(i)
                # Swap pickup and drop off locations for task
                curr_dropoff = task.dropoff_point
                task.dropoff_point = task.pickup_point
                task.pickup_point = curr_dropoff
                task.is_at_dropoff = False

                self._ready_tasks.append(task)


if __name__ == "__main__":
    grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)

    ta = TaskAssigner(grid, task_frequency=2, is_printing=True)
    agents = generate_n_agents(5)

    for curr_t in range(10):
        ready_agent = get_ready_agent(agents)
        if ready_agent:
            ready_task = ta.get_ready_task()
            if ready_task:
                # assign task to agent
                pass

        ta.increment_timestep()

    print(grid)
