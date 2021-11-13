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

urg = Warehouse.UniformRandomGrid()

# CROSSOVER_TILE_SIZE = 5
# CROSSOVER_TILE_NO = 1


def mate(arr_1: np.ndarray, arr_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return arr_1, arr_2


def mutate(arr: np.ndarray) -> np.ndarray:
    return arr


def evaluate(values: np.ndarray, no_agents: int, no_timesteps: int):
    try:
        grid: List = urg.get_uniform_random_grid(OPT_GRID_SHAPE, NO_STORAGE_LOCS)# This is a list, why PyCharm...
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

        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            # decorate the functions with `ray_remote_put` which automatically
            # `ray.get` arguments that are `ObjectRef`s and `ray.put`s returned results
            mate_fn=ray_remote_puts(mate).remote,
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
        return [ray.put(urg.get_uniform_random_grid(OPT_GRID_SHAPE, NO_STORAGE_LOCS))
                for _ in range(self.hparams.population_size)]


def train(pop_size, n_generations, n_agents, n_timesteps,
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
        "cluster_node": cluster_node,
        "mate_func": "none"
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
                               mut_tile_size=1, mut_tile_no=1,  # Not used so it doesn't matter the value
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

    parse_args = "pop_size,n_generations,n_agents,n_timesteps," \
                 "cluster_node,run_notes,run_name," \
                 "log_interval,save_interval"
    parse_args = parse_args.split(",")

    for parse_arg in parse_args:
        parser.add_argument(parse_arg)
    args = parser.parse_args()

    pop_size = int(args.pop_size)
    n_generations = int(args.n_generations)

    n_agents = int(args.n_agents)
    n_timesteps = int(args.n_timesteps)

    run_notes = args.run_notes
    run_name = args.run_name
    cluster_node = args.cluster_node

    log_interval = int(args.log_interval)
    save_interval = int(args.save_interval)
    using_wandb = True

    train(pop_size, n_generations, n_agents, n_timesteps,
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
