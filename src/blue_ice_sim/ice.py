import argparse
import functools
import os
import sys
import multiprocessing
import random
from dataclasses import dataclass
from itertools import product, repeat
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import plotly.express as px

@dataclass
class Ice:
    adjacents: list[Self]
    # Any adjacents to a border or frozen has a chance to be frozen
    border: bool = False
    frozen: bool = False
    # Used for any water blocked from the sun. These will never freeze
    blocked: bool = False

    def __str__(self):
        if self.border:
            return "X"
        if self.blocked:
            return "W"
        if self.frozen:
            return "\u2588"

        return "\u2591"


class IceFarm:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("Size must be positive.")

        self._size = size
        self._count = 0
        self._eff_yield = self._get_eff_yield()
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

    @property
    def size(self) -> int:
        return self._size

    @property
    def eff_yield(self) -> int:
        return self._eff_yield

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_center_reached(self) -> bool:
        return any(
            self._grid[i][j].frozen for i, j in product(
                range((len(self._grid) - 1) // 2, (len(self._grid) + 3) // 2 + 1),
                range((len(self._grid) - 1) // 2, (len(self._grid) + 3) // 2 + 1))
        )

    def print_adjacency(self):
        for row in self._grid:
            print(" ".join("X" if cell.border else str(len(cell.adjacents)) for cell in row))

    def update(self):
        # Treat the grid layout as Minecraft chunks, where each chunk has an independent weather
        # check
        for c_i, c_j in product(range(self._chunk_count), repeat=2):
            if random.random() < WEATHER_UPDATE_CHANCE:
                # Choose a random coordinate to weather update
                i = random.randint(1, CHUNK_SIZE) + (c_i * CHUNK_SIZE)
                j = random.randint(1, CHUNK_SIZE) + (c_j * CHUNK_SIZE)

                if (0 < i < self._size + 2 and 0 < j < self._size + 2
                        and not(
                            self._grid[i][j].frozen
                            or self._grid[i][j].border
                            or self._grid[i][j].blocked)):
                    if any(cell.frozen or cell.border for cell in self._grid[i][j].adjacents):
                        self._grid[i][j].frozen = True
                        self._count += 1

    def increment(self):
        water_cells = (
            cell for row in self._grid for cell in row
            if not (cell.frozen or cell.border or cell.blocked)
        )

        freezable_cells = [
            cell for cell in water_cells
            if any(adj for adj in cell.adjacents if adj.frozen or adj.border)
        ]

        if freezable_cells:
            block = random.choice(freezable_cells)
            block.frozen = True
            self._count += 1

    def _get_eff_yield(self):
        """
        The effective yield, the number of ice blocks that can exist.
        """
        eff_yield = self._size * (self._size - 1)
        # For farms crater than 7, another diagonal must be used that shouldn't be count
        if self._size > 7:
            eff_yield -= self._size
            # If odd, avoid counting the center twice when using a cross pattern
            if self._size % 2 == 1:
                eff_yield += 1

        return eff_yield

    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self._grid)

class Screen:
    def __init__(self, size: int, size_count: int, run_count: int, threshold: float = 1.0):
        self._farm = IceFarm(size)
        self._size_count = size_count
        self._run_count = run_count
        self._total_runs = self._size_count * self._run_count
        self._yield_threshold = self._farm.eff_yield * threshold
        self._yield = 0
        self._run = 0

    def update(self, current_size: int):
        self._run += 1
        completion_ratio = self._run / self._total_runs
        new_yield = int(self._yield_threshold * completion_ratio)

        if new_yield > self._yield:
            for _ in range(new_yield - self._yield):
                self._farm.increment()
                self.refresh(current_size)

            self._yield = new_yield

    def refresh(self, current_size: int):
        self._clear()
        print(self._farm, file=sys.stderr)
        print((
                f"{current_size} {(self._run % self._run_count) + 1}"
                f" {self._farm.count / self._farm.eff_yield:.1%}"
                f" {self._run / self._total_runs:.1%}"
            ),
            file=sys.stderr
        )

    @staticmethod
    def _clear():
        os.system("cls" if os.name == "nt" else "clear")


def simulate_generation(size: int, threshold: float = 1.0) -> int:
    farm = IceFarm(size)
    eff_yield = farm.eff_yield
    ticks = 0

    while farm.count < eff_yield * threshold:
        ticks += 1
        farm.update()

    return ticks


def simulate_center(size: int) -> float:
    farm = IceFarm(size)
    ticks = 0

    while farm.count < farm.eff_yield:
        ticks += 1
        farm.update()

        if farm.is_center_reached:
            return farm.count / farm.eff_yield

    return farm.count / farm.eff_yield


def sim_gen_moments(size: int, threshold: float = 1.0) -> list[int]:
    farm = IceFarm(size)
    eff_yield = farm.eff_yield
    ticks = 0
    moments = []

    last_count = curr_count = farm.count
    while curr_count < eff_yield * threshold:
        ticks += 1
        farm.update()
        curr_count = farm.count

        if curr_count > last_count:
            last_count = curr_count
            moments.append(ticks)

    return [moment / ticks for moment in moments]

CHUNK_SIZE = 16
# Each block has a 1 in 16th chance of being weather updated
WEATHER_UPDATE_CHANCE = 1 / 16
MIN_SIZE = 3

DEFAULT_SIZE = 7
DEFAULT_THRESHOLD = 0.95
DEFAULT_RUN_COUNT = 1000

def main():
    parser = argparse.ArgumentParser(
        description="""
Simulates the generation of a Minecraft ice farm and outputs a
comma-separated line for each size, minimum duration, maximum duration,
average duration, and median duration (all in ticks). If the size is
greater than 7, then an X pattern is used for the water sources;
otherwise, only a single diagnal is used.
""",
        epilog="""
Copyright (c) 2021 Dekameter <dekameter@giant.ink>

This work is licensed under the terms of the Artistic License 2.0.
For a copy, see <https://opensource.org/license/artistic-2-0>.
"""
    )
    parser.add_argument(
        "-o", "--out", default=None, type=Path,
        help=""""
Provide a path to output results to a text file. If not provided, then no file is written to and
the recorded statistics are discarded.
"""
    )
    parser.add_argument(
        "-s", "--size", default=DEFAULT_SIZE, type=int,
        help=(
            f"The size to test against, or if '--increment' is set,"
            f" run up to this size. Cannot be below {MIN_SIZE}, and"
            f" defaults to {DEFAULT_SIZE}."
        )
    )
    parser.add_argument(
        "-t", "--threshold", default=None, type=float,
        help=(
            f"Set a percentage to cutoff from 0.0 to 1.0. Defaults to"
            f" {DEFAULT_THRESHOLD} (aka. every block is filled). This"
            f" value is ignored when using --center."
        )
    )
    parser.add_argument(
        "-r", "--run", default=DEFAULT_RUN_COUNT, type=int,
        help=(
            f"Set how times to run the simulation for to more"
            f" accurately determine the statistics. Default to"
            f" {DEFAULT_RUN_COUNT} runs."
        )
    )
    parser.add_argument(
        "-i", "--increment", action="store_true",
        help="Increments starting at a size of 3 up to '--size' set."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--center", action="store_true",
        help="""
Instead of the standard mode, runs the simulation until the center is
reached. The minimum, maximum, average, and median yield percentages
are saved to the file. --threshold is ignored.
"""
    )
    group.add_argument(
        "--moment", action="store_true",
        help="""
Instead of the standard mode, computes the average time taken for each
increasing yield. After every size is simulated, it will then plot each
size with percentage of yield vs. percentage of time taken 
"""
    )

    args = parser.parse_args()

    out_path = args.out
    max_size = args.size
    threshold = args.threshold
    run_count = args.run
    increment = args.increment
    center_mode = args.center
    moment_mode = args.moment

    if max_size < MIN_SIZE:
        raise ValueError(f"--size must be at least {MIN_SIZE}.")

    if threshold is None:
        if center_mode:
            print("Warning: --center set, threshold will be ignored", file=sys.stderr)
            threshold = 1.0
        else:
            threshold = DEFAULT_THRESHOLD

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("--threshold must be inclusively between 0.0 and 1.0.")

    sizes = list(range(MIN_SIZE if increment else max_size, max_size + 1))
    if moment_mode:
        time_ratios = []

    screen = Screen(17, len(sizes), run_count, threshold)
    screen.refresh(sizes[0])

    out_path.unlink(True)
    out_path.touch()

    for size in sizes:
        if center_mode:
            with multiprocessing.Pool() as pool:
                runs = []
                for run in pool.imap_unordered(simulate_center, repeat(size, run_count)):
                    runs.append(run)
                    screen.update(size)

            if out_path is not None:
                with out_path.open("a") as fd:
                    fd.write((
                        f"{size},{min(runs)},{max(runs)},{sum(runs) / len(runs)}"
                        f",{list(sorted(runs))[len(runs) // 2]}\n"
                    ))
        elif moment_mode:
            with multiprocessing.Pool() as pool:
                runs = []
                for run in pool.imap_unordered(
                        functools.partial(sim_gen_moments, size),
                        repeat(threshold, run_count)):
                    runs.append(run)
                    screen.update(size)

                time_ratio = list(map(lambda x: sum(x) / len(x), zip(*runs)))
                time_ratios.append(time_ratio)
        else:
            with multiprocessing.Pool() as pool:
                runs = []
                for run in pool.imap_unordered(
                        functools.partial(simulate_generation, size),
                        repeat(threshold, run_count)):
                    runs.append(run)
                    screen.update(size)

            # Print the field size, min, max, average, and median tick runtime
            if out_path is not None:
                with out_path.open("a") as fd:
                    fd.write((
                        f"{size},{min(runs)},{max(runs)},{sum(runs) / len(runs)}"
                        f",{list(sorted(runs))[len(runs) // 2]}\n"
                    ))

    if moment_mode:
        yield_ratios = []
        for moments in time_ratios:
            yield_ratio = [i / len(moments) for i in range(len(moments))]
            yield_ratios.append(yield_ratio)

        repeated_sizes = np.hstack([
            list(repeat(size, len(ratio))) for size, ratio in zip(sizes, yield_ratios)
        ])
        time_ratios = np.hstack(time_ratios)
        yield_ratios = np.hstack(yield_ratios)

        df = pd.DataFrame.from_records({
            "size": repeated_sizes,
            "yield_ratio": yield_ratios,
            "time_ratio": time_ratios
        })

        fig = px.line(
            df, color="size", markers=True,
            x="time_ratio", y="yield_ratio"
        )
        fig.update_layout(
            xaxis = {
                "title": "Time Passed (%)",
                "tickformat": ",.1%"
            },
            yaxis = {
                "title": "Yield (%)",
                "tickformat": ",.1%"
            }
        )
        fig.show()

if __name__ == "__main__":
    main()
