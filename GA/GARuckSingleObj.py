import time
import pickle
from typing import List, Tuple, Optional, Set, Dict, Callable
import os

from numba import njit, jit
import numpy as np
import pandas as pd

from functools import partial
import wandb

import ruck
# from ruck.external.deap import select_nsga2
from ruck import R, Trainer
from ruck.external.ray import *

from Benchmark import Warehouse
# from MAPD.TokenPassing import TokenPassing
from Grid.GridWrapper import get_no_unreachable_locs
from Visualisations.Vis import VisGrid

opt_grid_start_x = 6
NO_STORAGE_LOCS = 560
OPT_GRID_SHAPE = (22, 44)
NO_LOCS = OPT_GRID_SHAPE[0] * OPT_GRID_SHAPE[1]


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
        values = values.astype(int)
        reachable_locs = get_no_unreachable_locs(values)
        # no_unreachable_locs = tp.get_no_unreachable_locs()
    except Exception as e:  # Bad way to do this but I do not want this to crash after 5hrs of training from a random edge case
        # tasks_completed = 0
        # no_unreachable_locs = 100000
        reachable_locs = 0
        print(f"Exception occurred: {e}")

    return reachable_locs


class WarehouseGAModule(ruck.EaModule):
    def __init__(
            self,
            population_size: int = 300,
            no_agents=5,
            no_timesteps=500,
            offspring_num: int = None,  # offspring_num (lambda) is automatically set to population_size (mu) when `None`
            member_size: int = 100,
            p_mate: float = 0.5,
            p_mutate: float = 0.5,
            ea_mode: str = 'mu_plus_lambda',
            log_interval: int = -1,
            save_interval: int = -1,
            no_generations: int = 0,
            pop_save_dir: str = ""
    ):
        self._population_size = population_size
        self.save_hyperparameters()
        self.eval_count = 0
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.no_generations = no_generations
        self.pop_save_dir = pop_save_dir

        # self.train_loop_func = partial(train_loop_func, self)

        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            # decorate the functions with `ray_remote_put` which automatically
            # `ray.get` arguments that are `ObjectRef`s and `ray.put`s returned results
            mate_fn=ray_remote_puts(mate).remote,
            mutate_fn=ray_remote_put(mutate).remote,
            # efficient to compute locally
            select_fn=functools.partial(R.select_tournament, k=3),
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
            # ENABLE multiprocessing
            map_fn=ray_map,
        )

        # def _eval(values):
        #     return evaluate(values, no_agents, no_timesteps)

        # eval_partial = partial()
        # eval function, we need to cache it on the class to prevent
        # multiple calls to ray.remote. We use ray.remote instead of
        # ray_remote_put like above because we want the returned values
        # not object refs to those values.
        self._ray_eval = ray.remote(evaluate).remote

    def evaluate_values(self, values):
        out = ray_map(self._ray_eval, values)

        if self.log_interval > -1 and (self.eval_count == 0 or (self.eval_count + 1) % self.log_interval == 0 or self.eval_count + 1 == self.no_generations):
            # no_locs = OPT_GRID_SHAPE[0] * OPT_GRID_SHAPE[1]
            # data = [[x, y/NO_LOCS] for (x, y) in out]
            data = [[x/NO_LOCS] for x in out]
            table = wandb.Table(data=data, columns=["no_reachable_locs"])
            gen = self.eval_count+1
            wandb.log({'my_histogram': wandb.plot.histogram(table, "no_reachable_locs",
                                                            title=f"Generation = {gen} Histogram of Percentage of Locs Reachable")})
            # table = wandb.Table(data=data, columns=["unique_tasks_completed", "perc_reachable_locs"])
            # wandb.log({f"gen_{self.eval_count+1}_tc_vs_ul": wandb.plot.scatter(table, "unique_tasks_completed", "perc_reachable_locs",
            #                                                                    title=f"Generation = {gen} Unique Tasks Completed Vs Percentage of Locs Reachable")})

        if self.log_interval > -1:
            log_dict = {
                "generation": self.eval_count,
                "no_reachable_locs_max": np.max(out),
                "no_reachable_locs_mean": np.mean(out)
            }
            wandb.log(log_dict)

        if self.save_interval > -1 and (self.eval_count == 0 or (self.eval_count + 1) % self.save_interval == 0 or self.eval_count + 1 == self.no_generations):
            # data = [[x, y] for (x, y) in out]
            val_data = list(zip(values, out))

            # TEMP: Need to change this for when on cluster
            file_name = os.path.join(wandb.run.dir, f"pop_{self.eval_count+1}.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(val_data, f)

            wandb.save(file_name)

        self.eval_count += 1
        return out

    # def generate_offspring(self, population):
    #     pass

    def gen_starting_values(self):
        urg = Warehouse.UniformRandomGrid()
        return [ray.put(urg.get_uniform_random_grid(OPT_GRID_SHAPE, NO_STORAGE_LOCS))
                for _ in range(self.hparams.population_size)]


def main():
    # initialize ray to use the specified system resources
    ray.shutdown()
    ray.init()

    # create and train the population
    pop_size = 128  # 0
    n_generations = 100  # 0
    no_agents = 5
    no_timesteps = 500

    config = {
        "pop_size": pop_size,
        "n_generations": n_generations,
        "no_agents": no_agents,
        "no_timesteps": no_timesteps,
        "fitness": "no_reachable_locs",
        "notes": "Test to see if working"
    }
    wandb.init(project="GARuck", entity="simonrosen42", config=config)

    # define our custom x axis metric
    wandb.define_metric("generation")
    # define which metrics will be plotted against it
    wandb.define_metric("no_reachable_locs_max", step_metric="generation")
    wandb.define_metric("no_reachable_locs_mean", step_metric="generation")

    module = WarehouseGAModule(population_size=pop_size, no_generations=n_generations, no_agents=no_agents, no_timesteps=no_timesteps,
                               log_interval=5, save_interval=5)
    trainer = Trainer(generations=n_generations, progress=True, is_saving=False, file_suffix="populations/pop")
    pop, logbook, halloffame = trainer.fit(module)
    # pop_vals = [member.value for member in pop]

    # with open("stats/hist.pkl", "wb") as f:
    #     pickle.dump(logbook.history, f)
    #
    # with open("populations/pop_final.pkl", "wb") as f:
    #     vals = [ray.get(member.value) for member in pop]
    #     pickle.dump(vals, f)

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


if __name__ == "__main__":
    # graph_fitnesses()
    # animate_grid()
    # draw_grids()
    main()
    # alt()
    # test()


if __name__ == "__main__":
    main()
