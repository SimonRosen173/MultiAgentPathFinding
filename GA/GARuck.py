import pickle
import time
from typing import List, Tuple, Optional, Set, Dict

from numba import njit, jit
import numpy as np
import pandas as pd

import ruck
from ruck.external.deap import select_nsga2
from ruck import *
from ruck.external.ray import *

from Benchmark import Warehouse
from MAPD.TokenPassing import TokenPassing
from Visualisations.Vis import VisGrid

opt_grid_start_x = 6
NO_STORAGE_LOCS = 560
OPT_GRID_SHAPE = (22, 44)


@njit
def mutate_flip_bits(a: np.ndarray, p: float = 0.05) -> np.ndarray:
    return a ^ (np.random.random(a.shape) < p)


@njit
def one_point_crossover_2d(arr_1: np.ndarray, arr_2: np.ndarray,
                           crossover_point: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    c_pt = crossover_point
    # print(f"arr_1 no of storage locs: {np.count_nonzero(arr_1)}")
    # print(f"arr_2 no of storage locs: {np.count_nonzero(arr_2)}")
    c_arr_1 = arr_1.copy()
    c_arr_2 = arr_2.copy()

    tl_slices = (slice(None, c_pt[0]), slice(None, c_pt[0]))
    tr_slices = (slice(None, c_pt[0]), slice(c_pt[0], None))
    bl_slices = (slice(c_pt[0], None), slice(None, c_pt[0]))
    br_slices = (slice(c_pt[0], None), slice(c_pt[0], None))

    slices = None
    rand_quad = np.random.randint(0, 4)  # Randomly choose which quadrant to do crossover off
    if rand_quad == 0:
        slices = tl_slices
    elif rand_quad == 1:
        slices = tr_slices
    elif rand_quad == 2:
        slices = bl_slices
    else:
        slices = br_slices

    c_arr_1[slices] = arr_2[slices].copy()
    c_arr_2[slices] = arr_1[slices].copy()

    # print(f"c_arr_1 no of storage locs: {np.count_nonzero(c_arr_1)}")
    # print(f"c_arr_2 no of storage locs: {np.count_nonzero(c_arr_2)}")

    return c_arr_1, c_arr_2


@njit
def regain_no_storage_locs(arr: np.ndarray) -> np.ndarray:
    no_storage_locs = np.count_nonzero(arr)

    if no_storage_locs < NO_STORAGE_LOCS:
        while no_storage_locs < NO_STORAGE_LOCS:
            y = np.random.randint(0, arr.shape[0])
            x = np.random.randint(0, arr.shape[1])
            if arr[y, x] == 0:
                arr[y, x] = 1
                no_storage_locs += 1
    elif no_storage_locs > NO_STORAGE_LOCS:
        while no_storage_locs > NO_STORAGE_LOCS:
            y = np.random.randint(0, arr.shape[0])
            x = np.random.randint(0, arr.shape[1])
            if arr[y, x] == 1:
                arr[y, x] = 0
                no_storage_locs -= 1

    return arr


# @njit
def mate(arr_1: np.ndarray, arr_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    static_arr_1 = arr_1[:, :opt_grid_start_x]
    static_arr_2 = arr_2[:, :opt_grid_start_x]
    opt_arr_1 = arr_1[:, opt_grid_start_x:]
    opt_arr_2 = arr_2[:, opt_grid_start_x:]

    c_pt = (opt_arr_1.shape[0]//2, opt_arr_1.shape[1]//2)
    opt_c_arr_1, opt_c_arr_2 = one_point_crossover_2d(opt_arr_1, opt_arr_2, c_pt)

    # Can't use concatenate cause of njit
    new_shape = (static_arr_1.shape[0], static_arr_1.shape[1] + opt_c_arr_1.shape[1])
    c_arr_1 = np.zeros(new_shape)
    c_arr_2 = np.zeros(new_shape)
    c_arr_1[:, :static_arr_1.shape[1]] = static_arr_1
    c_arr_1[:, static_arr_1.shape[1]:] = opt_c_arr_1
    c_arr_2[:, :static_arr_1.shape[1]] = static_arr_2
    c_arr_2[:, static_arr_1.shape[1]:] = opt_c_arr_2
    # c_arr_1 = np.concatenate([static_arr_1, opt_c_arr_1], axis=1)
    # c_arr_2 = np.concatenate([static_arr_2, opt_c_arr_2], axis=1)
    # return c_arr_1, c_arr_2
    return arr_1, arr_2


# @njit
def mutate(arr: np.ndarray) -> np.ndarray:
    static_arr = arr[:, :opt_grid_start_x]
    opt_arr = arr[:, opt_grid_start_x:]

    opt_arr = opt_arr.astype(bool)
    opt_arr = mutate_flip_bits(opt_arr)
    opt_arr = opt_arr.astype(int)

    # Need to retain number of storage locs
    opt_arr = regain_no_storage_locs(opt_arr)

    # Can't use concatenate cause of njit
    new_shape = (static_arr.shape[0], static_arr.shape[1] + opt_arr.shape[1])
    arr = np.zeros(new_shape)
    arr[:, :static_arr.shape[1]] = static_arr
    arr[:, static_arr.shape[1]:] = opt_arr

    return arr


def evaluate(values: np.ndarray):
    try:
        # noinspection PyTypeChecker
        grid: List = values.tolist()  # This is a list, why PyCharm...
        # print("Evaluating... ")
        y_len = len(grid)
        x_len = len(grid[0])
        no_agents = 5
        max_t = 250

        non_task_endpoints = [(y, 0) for y in range(y_len)]
        start_locs = non_task_endpoints[:no_agents]

        tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, max_t, task_frequency=1,
                          is_logging_collisions=True)
        final_agents = tp.compute()
        tasks_completed = tp.get_no_tasks_completed()
        no_unreachable_locs = tp.get_no_unreachable_locs()
    except Exception as e:  # Bad way to do this but I do not want this to crash after 5hrs of training from a random edge case
        print(f"Exception occurred: {e}")
        tasks_completed = 0
        no_unreachable_locs = 100000

    # Maximising tasks_completed and minimising no_unreachable_locs
    return tasks_completed, -1 * no_unreachable_locs
    # return 1, 1


class WarehouseGAModule(ruck.EaModule):
    def __init__(
            self,
            population_size: int = 300,
            offspring_num: int = None,  # offspring_num (lambda) is automatically set to population_size (mu) when `None`
            member_size: int = 100,
            p_mate: float = 0.5,
            p_mutate: float = 0.5,
            ea_mode: str = 'mu_plus_lambda'
    ):
        self._population_size = population_size
        self.save_hyperparameters()
        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            # decorate the functions with `ray_remote_put` which automatically
            # `ray.get` arguments that are `ObjectRef`s and `ray.put`s returned results
            mate_fn=ray_remote_puts(mate).remote,
            mutate_fn=ray_remote_put(mutate).remote,
            # efficient to compute locally
            select_fn=select_nsga2,  # Does this work?
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
            # ENABLE multiprocessing
            map_fn=ray_map,
        )
        # eval function, we need to cache it on the class to prevent
        # multiple calls to ray.remote. We use ray.remote instead of
        # ray_remote_put like above because we want the returned values
        # not object refs to those values.
        self._ray_eval = ray.remote(evaluate).remote

    def evaluate_values(self, values):
        return ray_map(self._ray_eval, values)

    # def generate_offspring(self, population):
    #     pass

    def gen_starting_values(self):
        urg = Warehouse.UniformRandomGrid()
        return [ray.put(urg.get_uniform_random_grid(OPT_GRID_SHAPE, NO_STORAGE_LOCS))
                for _ in range(self.hparams.population_size)]


def main():
    # initialize ray to use the specified system resources
    ray.init()

    # create and train the population
    pop_size = 160  # 0
    n_generations = 1000  # 0
    module = WarehouseGAModule(population_size=pop_size)
    trainer = Trainer(generations=n_generations, progress=True, is_saving=True, file_suffix="populations/pop", save_interval=10)
    pop, logbook, halloffame = trainer.fit(module)
    # pop_vals = [member.value for member in pop]

    with open("stats/hist.pkl", "wb") as f:
        pickle.dump(logbook.history, f)

    with open("populations/pop_final.pkl", "wb") as f:
        vals = [ray.get(member.value) for member in pop]
        pickle.dump(vals, f)

    # # Not the best way to choose 'best' individuals but should give a reasonable idea of how well GA is performing
    # sorted_members = sorted(pop, key=lambda x: x.fitness[0] + x.fitness[1])
    # for i in range(5):
    #     curr_grid = ray.get(sorted_members[i].value)
    #     vis = VisGrid(curr_grid, (800, 400), 25, tick_time=0.2)
    #     vis.save_to_png(f"best/best_grid_{i}")
    #     vis.window.close()

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    # print('best member:', halloffame.members[0])

    # best_member = halloffame.members[0]
    # obj_ref = best_member.value
    # best_grid = ray.get(obj_ref)
    #
    # for i, member in enumerate(halloffame.members):
    #     curr_grid = ray.get(best_member.value)
    #     vis = VisGrid(curr_grid, (800, 400), 25, tick_time=0.2)
    #     vis.save_to_png(f"best/best_grid_{i}")
    #     vis.window.close()
    # print(type())


def test():
    grid_1 = Warehouse.get_uniform_random_grid(OPT_GRID_SHAPE, NO_STORAGE_LOCS)
    grid_2 = Warehouse.get_uniform_random_grid(OPT_GRID_SHAPE, NO_STORAGE_LOCS)

    vis_1 = VisGrid(grid_1, (800, 400), 25, tick_time=0.2)
    vis_1.save_to_png("grid_1")
    vis_1.window.close()

    vis_2 = VisGrid(grid_2, (800, 400), 25, tick_time=0.2)
    vis_2.save_to_png("grid_2")
    vis_2.window.close()

    np_grid_1 = np.array(grid_1)
    np_grid_2 = np.array(grid_2)

    # start = time.time()
    # for _ in range(1):
    #     c_grid_1, c_grid_2 = mate(np_grid_1, np_grid_2)
    # print(f"Time elapsed: {time.time() - start}")

    c_grid_1, c_grid_2 = mate(np_grid_1, np_grid_2)
    c_vis_1 = VisGrid(c_grid_1, (800, 400), 25, tick_time=0.2)
    c_vis_1.save_to_png("c_grid_1")
    c_vis_1.window.close()

    c_vis_2 = VisGrid(c_grid_2, (800, 400), 25, tick_time=0.2)
    c_vis_2.save_to_png("c_grid_2")
    c_vis_2.window.close()

    mut_grid = mutate(np_grid_1)
    mut_vis = VisGrid(mut_grid, (800, 400), 25, tick_time=0.2)
    mut_vis.save_to_png("mut_vis")
    mut_vis.window.close()

    pass


def graph_fitnesses():
    for pop_name in ["final", "0", "500"]:
        pop = None
        with open(f"populations/pop_{pop_name}.pkl", "rb") as f:
            pop = pickle.load(f)

        tasks_completed_arr = []
        no_unreachable_locs_arr = []
        print(f"Re-evaluating fitnesses for {pop_name}...")

        if pop is not None:
            for i, member in enumerate(pop):
                print(f"Re-evaluating {i+1}/{len(pop)}...")
                tasks_completed, no_unreachable_locs = evaluate(member)
                no_unreachable_locs *= -1
                tasks_completed_arr.append(tasks_completed)
                no_unreachable_locs_arr.append(no_unreachable_locs)
                # if i % 10 == 0:
                #     vis_1 = VisGrid(member, (800, 400), 25, tick_time=0.2)
                #     vis_1.save_to_png(f"final_grids/grid_{i}")
                #     vis_1.window.close()

        df_dict = {"tasks_completed":tasks_completed_arr, "no_unreachable_locs":no_unreachable_locs_arr}
        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(f"csvs/{pop_name}.csv")


def draw_grids():
    pop = None
    with open("populations/pop_final.pkl", "rb") as f:
        pop = pickle.load(f)

    if pop is not None:
        vis_1 = VisGrid(pop[103], (800, 400), 25, tick_time=0.2)
        vis_1.save_to_png(f"best/final_103")
        vis_1.window.close()


def animate_grid():
    pop = None
    with open("populations/pop_final.pkl", "rb") as f:
        pop = pickle.load(f)

    good_member = pop[26]
    bad_member = pop[103]
    grid = good_member

    y_len = len(grid)
    x_len = len(grid[0])

    no_agents = 5
    max_t = 250

    non_task_endpoints = [(y, 0) for y in range(y_len)]
    start_locs = non_task_endpoints[:no_agents]

    tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, max_t, task_frequency=1,
                      is_logging_collisions=True)
    final_agents = tp.compute()

    vis = VisGrid(grid, (800, 400), 25, tick_time=0.2)
    vis.window.getMouse()
    vis.animate_mapd(final_agents, is_pos_xy=False)
    vis.window.getMouse()


def alt():
    pop = None
    with open("populations/pop_final.pkl", "rb") as f:
        pop = pickle.load(f)

    vis_1 = VisGrid(pop[0], (800, 400), 25, tick_time=0.2)
    vis_1.save_to_png("grid_save")
    vis_1.window.close()


if __name__ == "__main__":
    # graph_fitnesses()
    animate_grid()
    # draw_grids()
    # main()
    # alt()
    # test()
