import argparse
import os
import sys
import multiprocessing
import random
from itertools import product, repeat
from dataclasses import dataclass
from typing import List, Self

@dataclass
class Ice:
    adjacents: List[Self]
    # Any adjacents to a border or frozen has a chance to be frozen
    border: bool = False
    frozen: bool = False
    # Used for any water blocked from the sun. These will never freeze
    blocked: bool = False

    def __str__(self):
        if self.border:
            return 'X'
        elif self.blocked:
            return 'W'
        elif self.frozen:
            return '\u2588'
        else:
            return '\u2591'


class IceFarm:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("Size must be positive.")

        self._size = size
        # Ceiling divide
        self._chunk_count = size // CHUNK_SIZE + (0 if size % CHUNK_SIZE == 0 else 1)
        self._grid = [[Ice([]) for _ in range(size + 2)] for _ in range(size + 2)]

        for i in range(size + 2):
            for j in [0, size + 1]:
                self._grid[i][j].border = True
        
        for i in [0, size + 1]:
            for j in range(size + 2):
                self._grid[i][j].border = True
        
        for i in range(1, size + 1):
            self._grid[i][i].blocked = True
        
        if size > 7:
            for i in range(1, size + 1):
                self._grid[i][self._size + 1 - i].blocked = True

        for i in range(1, size + 1):
            for j in range(1, size + 1):
                # Ignores the diagonals since experimental evidence supports this
                for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                    if i != x or j != y:
                        self._grid[i][j].adjacents.append(self._grid[x][y])

    def print_adjacency(self):
        for row in self._grid:
            print(' '.join('X' if cell.border else str(len(cell.adjacents)) for cell in row))
    
    def __str__(self):
        return '\n'.join(' '.join(str(cell) for cell in row) for row in self._grid)
    
    def update(self):
        # Treat the grid layout as Minecraft chunks, where each chunk has an independent weather check
        for c_i, c_j in product(range(self._chunk_count), repeat=2):
            if random.random() < WEATHER_UPDATE_CHANCE:
                # Choose a random coordinate to weather update
                i = random.randint(1, CHUNK_SIZE) + (c_i * CHUNK_SIZE)
                j = random.randint(1, CHUNK_SIZE) + (c_j * CHUNK_SIZE)

                if 0 < i < self._size + 2 and 0 < j < self._size + 2 and not self._grid[i][j].border and not self._grid[i][j].blocked:
                    if any(cell.frozen or cell.border for cell in self._grid[i][j].adjacents):
                        self._grid[i][j].frozen = True

    def count(self):
        """
        The number of ice blocks that currently exist.
        """
        return sum(1 for i, j in product(range(self._size + 2), repeat=2) if not self._grid[i][j].border and self._grid[i][j].frozen)
    
    def get_eff_yield(self):
        """
        The effective yield, the number of ice blocks that can exist.
        """
        eff_yield = self._size * (self._size - 1)
        if self._size > 7:
            eff_yield -= self._size
            # If odd, avoid counting the center twice when using a cross pattern
            if self._size % 2 == 1:
                eff_yield += 1

        
        return eff_yield
    
    def center_touched(self):
        # print([(i, j) for i, j in product(range((len(self._grid) - 1) // 2, (len(self._grid) + 3) // 2 + 1), range((len(self._grid) - 1) // 2, (len(self._grid) + 3) // 2 + 1))])
        # print(len(self._grid))
        # sys.exit()

        return any(self._grid[i][j].frozen for i, j in product(range((len(self._grid) - 1) // 2, (len(self._grid) + 3) // 2 + 1), range((len(self._grid) - 1) // 2, (len(self._grid) + 3) // 2 + 1)))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def update_screen(farm: IceFarm, ticks: int):
    clear_screen()
    print(farm)
    print(size, farm.count(), farm.get_eff_yield(), ticks)
    

def simulate_generation(size: int, debug: bool = False) -> int:
    farm = IceFarm(size)
    eff_yield = farm.get_eff_yield()
    ticks = 0

    if debug:
        update_screen(farm, ticks)

    last_count = curr_count = farm.count()
    while curr_count < eff_yield * 0.9:
    # while curr_count < eff_yield:
        ticks += 1
        farm.update()
        curr_count = farm.count()

        if debug and curr_count > last_count:
            last_count = curr_count
            update_screen(farm, ticks)
    
    return ticks

def simulate_center(size: int, debug: bool = False) -> int:
    farm = IceFarm(size)
    eff_yield = farm.get_eff_yield()
    ticks = 0

    if debug:
        update_screen(farm, ticks)

    last_count = curr_count = farm.count()
    while curr_count < eff_yield:
        ticks += 1
        farm.update()
        curr_count = farm.count()

        if debug and curr_count > last_count:
            last_count = curr_count
            update_screen(farm, ticks)
        
        if farm.center_touched():
            return farm.count() / farm.get_eff_yield()
    
    return farm.count() / farm.get_eff_yield()

def sim_gen_moments(size: int) -> list[int]:
    farm = IceFarm(size)
    eff_yield = farm.get_eff_yield()
    ticks = 0
    moments = []

    last_count = curr_count = farm.count()
    while curr_count < eff_yield:
        ticks += 1
        farm.update()
        curr_count = farm.count()

        if curr_count > last_count:
            last_count = curr_count
            moments.append(ticks)
    
    return [moment / ticks for moment in moments]

# Print the field size, min, max, average, and median tick runtime

CHUNK_SIZE = 16
# Each block has a 1 in 16th chance of being weather updated
WEATHER_UPDATE_CHANCE = 1 / 16
RUN_COUNT = 1000
MIN_SIZE = 3

def main():
    parser = argparse.ArgumentParser(
        description="""
Simulates the generation of a Minecraft ice farm and outputs a comma-separated line for each size, minimum duration,
maximum duration, average duration, and median duration (all in ticks). If the size is greater than 7, than an X pattern is used for
the water sources; otherwise, only a single diagnal is used.
""",
        epilog="""
Copyright (c) 2021 Dekameter <dekameter@giant.ink>

This work is licensed under the terms of the Artistic License 2.0.  
For a copy, see <https://opensource.org/license/artistic-2-0>.
""",
    )
    parser.add_argument("-s", "--size", default=7, type=int,
                        help="The size to test against, or if '--increment' is set, run up to this size. Cannot be below {},"
                             " and defaults to 7.".format(MIN_SIZE))
    parser.add_argument("-i", "--increment", action="store_true",
                        help="Increments starting at a size of 3 up to '--size' set.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run in debug mode, where the farm will be printed instead, and no parallelism is done.")
    
    args = parser.parse_args()

    max_size = args.size
    increment = args.increment
    debug = args.debug

    if max_size < 3:
        raise ValueError("Size must be at least {}.".format(MIN_SIZE))

    sizes = range(27 if increment else max_size, max_size + 1)

    for size in sizes:
        # If we're running in debug mode, we don't want multiprocessing, and only do one run for each
        if debug:
            simulate_generation(size, debug)
        else:
            with multiprocessing.Pool() as pool:
                runs = list(pool.imap_unordered(simulate_generation, repeat(size, RUN_COUNT)))
        
            print("{},{},{},{},{}".format(size, min(runs), max(runs), sum(runs) / len(runs), list(sorted(runs))[len(runs) // 2]))

        # with multiprocessing.Pool() as pool:
        #     runs = list(pool.imap_unordered(sim_gen_moments, repeat(size, RUN_COUNT)))
        #     moments = list(map(lambda x: sum(x) / len(x), zip(*runs)))

        # print("{},{}".format(size, ",".join(str(moment) for moment in moments)))


if __name__ == "__main__":
    main()
