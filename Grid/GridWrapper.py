from Grid import grid_fast
import numpy as np
from Benchmark import Warehouse

if __name__ == "__main__":
    # grid = Warehouse.txt_to_grid("map_warehouse.txt", use_curr_workspace=True, simple_layout=False)
    # grid = np.array(grid)
    num_storage_locs = 560  # 560
    grid = Warehouse.get_uniform_random_grid((22, 44), num_storage_locs)
    grid = np.array(grid)
    # np_arr = np.ones((2,2), dtype=int)
    # grid_fast.get
    no_reachable_locs = grid_fast.get_no_reachable_locs(grid, 0, 0)
    print(f"{no_reachable_locs}/{grid.shape[0] * grid.shape[1]}")


def get_no_unreachable_locs(grid: np.ndarray) -> int:
    return grid_fast.get_no_reachable_locs(grid)