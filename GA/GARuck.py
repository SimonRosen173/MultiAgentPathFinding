import time
import pickle
from typing import List, Tuple, Optional, Set, Dict, Callable
import os
import shutil
import argparse

from numba import njit, jit
import numpy as np
import pandas as pd

from functools import partial
import wandb

import ruck
from ruck.external.deap import select_nsga2
from ruck import R, Trainer
from ruck.external.ray import *

from Benchmark import Warehouse
from MAPD.TokenPassing import TokenPassing
from Grid.GridWrapper import get_no_unreachable_locs
from Visualisations.Vis import VisGrid

opt_grid_start_x = 6
NO_STORAGE_LOCS = 560
OPT_GRID_SHAPE = (22, 44)
NO_LOCS = OPT_GRID_SHAPE[0] * OPT_GRID_SHAPE[1]
STATIC_LOCS_NO = OPT_GRID_SHAPE[0] * opt_grid_start_x

# CROSSOVER_TILE_SIZE = 5
# CROSSOVER_TILE_NO = 1


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
def tiled_crossover(arr_1, arr_2, tile_size, no_tiles=1):
    assert arr_1.shape == arr_2.shape, "Arrays must be of same shape"

    max_y = arr_1.shape[0] - tile_size
    max_x = arr_1.shape[1] - tile_size
    new_arr_1 = arr_1.copy()
    new_arr_2 = arr_2.copy()

    for i in range(no_tiles):
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)
        slices = (slice(y, y+tile_size), slice(x, x+tile_size))

        new_arr_1[slices] = arr_2[slices].copy()
        new_arr_2[slices] = arr_1[slices].copy()

    return new_arr_1, new_arr_2


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


def mate(arr_1: np.ndarray, arr_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return mate_njit(arr_1, arr_2)


@njit
def mate_njit(arr_1: np.ndarray, arr_2: np.ndarray, tile_size, tile_no=1) -> Tuple[np.ndarray, np.ndarray]:
    static_arr_1 = arr_1[:, :opt_grid_start_x]
    static_arr_2 = arr_2[:, :opt_grid_start_x]
    opt_arr_1 = arr_1[:, opt_grid_start_x:]
    opt_arr_2 = arr_2[:, opt_grid_start_x:]

    # c_pt = (opt_arr_1.shape[0]//2, opt_arr_1.shape[1]//2)
    # opt_c_arr_1, opt_c_arr_2 = one_point_crossover_2d(opt_arr_1, opt_arr_2, c_pt)

    opt_c_arr_1, opt_c_arr_2 = tiled_crossover(opt_arr_1, opt_arr_2, tile_size, tile_no)

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


def evaluate(values: np.ndarray, no_agents: int, no_timesteps: int):
    try:
        grid: List = values.tolist()  # This is a list, why PyCharm...
        y_len = len(grid)
        x_len = len(grid[0])

        non_task_endpoints = [(0, y) for y in range(y_len)]
        start_locs = non_task_endpoints[:no_agents]

        tp = TokenPassing(grid, no_agents, start_locs, non_task_endpoints, no_timesteps, task_frequency=1,
                          is_logging_collisions=True)
        final_agents = tp.compute()
        unique_tasks_completed = tp.get_no_unique_tasks_completed()
        # print(unique_tasks_completed)
        no_unreachable_locs = tp.get_no_unreachable_locs()
        reachable_locs = NO_LOCS - no_unreachable_locs
    except Exception as e:  # Bad way to do this but I do not want this to crash after 5hrs of training from a random edge case
        print(f"Exception occurred: {e}")
        tasks_completed = 0
        reachable_locs = 0
        unique_tasks_completed = 0

    return unique_tasks_completed, reachable_locs


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
            mut_tile_no: int = 1,
            mut_tile_size: int = 5,
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
        self.mut_tile_no = mut_tile_no
        self.mut_tile_size = mut_tile_size

        # self.train_loop_func = partial(train_loop_func, self)
        def _mate(arr_1: np.ndarray, arr_2: np.ndarray):
            return mate_njit(arr_1, arr_2,
                             tile_size=self.mut_tile_size, tile_no=self.mut_tile_no)

        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            # decorate the functions with `ray_remote_put` which automatically
            # `ray.get` arguments that are `ObjectRef`s and `ray.put`s returned results
            mate_fn=ray_remote_puts(_mate).remote,
            mutate_fn=ray_remote_put(mutate).remote,
            # efficient to compute locally
            # select_fn=functools.partial(R.select_tournament, k=3),
            select_fn=select_nsga2,
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
            # ENABLE multiprocessing
            map_fn=ray_map,
        )

        def _eval(values):
            return evaluate(values, no_agents, no_timesteps)

        # eval_partial = partial()
        # eval function, we need to cache it on the class to prevent
        # multiple calls to ray.remote. We use ray.remote instead of
        # ray_remote_put like above because we want the returned values
        # not object refs to those values.
        self._ray_eval = ray.remote(_eval).remote

    def evaluate_values(self, values):
        out = ray_map(self._ray_eval, values)
        data = None

        # wandb Stuff
        if self.log_interval > -1 and (self.eval_count == 0 or (self.eval_count + 1) % self.log_interval == 0 or self.eval_count + 1 == self.no_generations):
            # no_locs = OPT_GRID_SHAPE[0] * OPT_GRID_SHAPE[1]
            # unique_tasks_completed, reachable_locs
            # data = [[x, y/NO_LOCS] for (x, y) in out]
            gen = self.eval_count+1
            data = [[x, y/NO_LOCS, gen] for (x, y) in out]
            table = wandb.Table(data=data, columns=["unique_tasks_completed", "perc_reachable_locs", "gen"])
            wandb.log({f'gen_{gen}_reach_locs_hist': wandb.plot.histogram(table, "perc_reachable_locs",
                                                            title=f"Generation = {gen} Histogram of Percentage of Locs Reachable")})
            wandb.log({f'gen_{gen}_unique_tasks_hist': wandb.plot.histogram(table, "unique_tasks_completed",
                                                                          title=f"Generation = {gen} Histogram of No of Unique Tasks Completed")})
            # table = wandb.Table(data=data, columns=["unique_tasks_completed", "perc_reachable_locs"])
            # wandb.log({f"gen_{self.eval_count+1}_tc_vs_ul": wandb.plot.scatter(table, "unique_tasks_completed", "perc_reachable_locs",
            #                                                                    title=f"Generation = {gen} Unique Tasks Completed Vs Percentage of Locs Reachable")})

        if self.log_interval > -1:
            reachable_locs = [y for (x, y) in out]
            unique_tasks_completed = [x for (x, y) in out]
            log_dict = {
                "generation": self.eval_count,
                "perc_reachable_locs_max": np.max(reachable_locs)/NO_LOCS,
                "perc_reachable_locs_mean": np.mean(reachable_locs)/NO_LOCS,
                "perc_reachable_locs_min": np.min(reachable_locs)/NO_LOCS,
                "perc_reachable_locs_var": np.var(reachable_locs)/NO_LOCS,

                "unique_tasks_completed_max": np.max(unique_tasks_completed)/NO_LOCS,
                "unique_tasks_completed_mean": np.mean(unique_tasks_completed)/NO_LOCS,
                "unique_tasks_completed_min": np.min(unique_tasks_completed)/NO_LOCS,
                "unique_tasks_completed_var": np.var(unique_tasks_completed)/NO_LOCS,
            }
            wandb.log(log_dict)
            # tbl_data = [[x, y, self.eval_count+1] for (x, y) in data]
            if data is None:
                gen = self.eval_count+1
                data = [[x, y/NO_LOCS, gen] for (x, y) in out]

            fitness_table = wandb.Table(columns=["unique_tasks_completed", "reachable_locs", "gen"], data=data)
            wandb.log({"fitness_table": fitness_table})

        if self.save_interval > -1 and (self.eval_count == 0 or (self.eval_count + 1) % self.save_interval == 0 or self.eval_count + 1 == self.no_generations):
            # data = [[x, y] for (x, y) in out]
            val_data = list(zip(values, out))

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


def train(pop_size, n_generations, n_agents,
          n_timesteps, mut_tile_size, mut_tile_no,
          using_wandb, log_interval, save_interval,
          cluster_node,
          run_notes, run_name):
    # initialize ray to use the specified system resources
    ray.init()

    # create and train the population
    # pop_size = 10  # 0
    # n_generations = 10  # 0
    # no_agents = 5
    # no_timesteps = 500
    # using_wandb = True
    # log_interval = 2
    # save_interval = 2
    # mut_tile_size = 5
    # mut_tile_no = 1

    config = {
        "pop_size": pop_size,
        "n_generations": n_generations,
        "no_agents": n_agents,
        "no_timesteps": n_timesteps,
        "fitness": "unique_tasks_completed, reachable_locs",
        "mut_tile_size": mut_tile_size,
        "mut_tile_no": mut_tile_no,
        "cluster_node": cluster_node,
        "mate_func": "tiled_crossover"
    }
    # notes = "Test to see if this works :)"
    if run_name == "":
        run_name = None

    if using_wandb:
        wandb.init(project="GARuck", entity="simonrosen42", config=config, notes=run_notes, name=run_name)

        # define our custom x axis metric
        wandb.define_metric("generation")
        # define which metrics will be plotted against it
        wandb.define_metric("perc_reachable_locs_max", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_mean", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_min", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_var", step_metric="generation")

        wandb.define_metric("perc_reachable_locs_max", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_mean", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_min", step_metric="generation")
        wandb.define_metric("perc_reachable_locs_var", step_metric="generation")

    else:
        log_interval = -1
        save_interval = -1

    module = WarehouseGAModule(population_size=pop_size,
                               no_generations=n_generations, no_agents=n_agents,
                               no_timesteps=n_timesteps,
                               mut_tile_size=mut_tile_size, mut_tile_no=mut_tile_no,
                               log_interval=log_interval, save_interval=save_interval)
    trainer = Trainer(generations=n_generations, progress=True)
    pop, logbook, halloffame = trainer.fit(module)

    # Clean Up Local Files
    # wandb_dir = wandb.run.dir
    # wandb_dir = wandb_dir.split("\\")[:-2]
    # wandb_dir = "/".join(wandb_dir)

    wandb.finish()
    # shutil.rmtree(wandb_dir)


if __name__ == "__main__":
    # train()
    parser = argparse.ArgumentParser()

    parse_args = "pop_size,n_generations,n_agents,n_timesteps,mut_tile_size,mut_tile_no," \
                 "cluster_node,run_notes,run_name," \
                 "log_interval,save_interval"
    parse_args = parse_args.split(",")

    for parse_arg in parse_args:
        parser.add_argument(parse_arg)
    args = parser.parse_args()

    pop_size = args.pop_size
    n_generations = args.n_generations

    n_agents = args.n_agents
    n_timesteps = args.n_timesteps

    mut_tile_size = args.mut_tile_size
    mut_tile_no = args.mut_tile_no

    run_notes = args.run_notes
    run_name = args.run_name
    cluster_node = args.cluster_node

    log_interval = args.log_interval
    save_interval = args.save_interval
    using_wandb = True

    train(pop_size, n_generations, n_agents,
          n_timesteps, mut_tile_size, mut_tile_no,
          using_wandb, log_interval, save_interval,
          cluster_node,
          run_notes, run_name)
    # using_wandb, log_interval, save_interval,
    # fitness_notes, mate_notes,
    # cluster_node,
    # run_notes, run_name
    # main()

    # TEST
    # pop_size,n_generations,n_agents,n_timesteps,mut_tile_size,mut_tile_no,cluster_node,run_notes,run_name
    # log_interval,save_interval

    # python GARuck.py 10 10 5 500 4 1 -1 "Test" "Test Run" 2 2
