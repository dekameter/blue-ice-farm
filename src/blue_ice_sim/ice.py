"""A script used to simulate and analyze the nature of Minecraft ice farm generation.
"""

import argparse
import functools
import os
import sys
import time
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
    """A single block of ice that tracks whether or not its frozen, blocked (from the sun), and
    whether or not it's a border block. A list of adjacents is also tracked.
    """
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
    """A representation of an optimal Minecraft ice farm that simulates the propogation of ice from
    the borders toward the center. The farm is appropriately split into chunks, and each chunk
    has the possibility of forming a single ice block if there is an adjacent solid or frozen block.
    """
    def __init__(self, size: int) -> None:
        """Generates an ice farm based on Minecraft guides and tutorials. The size given is the size
        in any one dimension, so the ice farm is a grid with size * size blocks. An extra one block
        border is formed around the ice farm, making the total grid size (size + 2)^2. For
        sizes <= 7, a single diagonal is used to block the ice so that water can reform the ice
        afterwards. For sizes > 7, two diagonals / an X is instead used, as that is necessary to
        refill the farm properly.

        :param size: _description_
        :raises ValueError: _description_
        :yield: _description_
        """
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
        """How large the ice farm is in one dimension. Excludes the border. In other words,
        size = 3 means a 3x3 farm with a 4x4 single block border all around.

        :return: The size of the farm in one dimension
        """
        return self._size

    @property
    def eff_yield(self) -> int:
        """The maximum possible frozen blocks for this farm based on its size

        :return: The max number of frozen blocks
        """
        return self._eff_yield

    @property
    def count(self) -> int:
        """The count of how many current blocks are frozen

        :return: How many blocks are frozen
        """
        return self._count

    @property
    def is_center_reached(self) -> bool:
        """Determines whether the ice has become adjacent to the center of the farm or not

        :return: True for ice adjacent to center, otherwise False
        """
        min_corner = (len(self._grid) // 2) - 1
        max_corner = ((len(self._grid) + 1) // 2) + 1

        if self._size > 7 and self._size % 2 == 0:
            min_corner -= 1
            max_corner += 1

        return any(
            self._grid[i][j].frozen for i, j in product(
                range(min_corner, max_corner), range(min_corner, max_corner))
        )

    def print_adjacency(self):
        """Instead of standard printing, it prints each ice block as the number of adjacents it
        has. Useful for debugging purposes.
        """
        for row in self._grid:
            print(" ".join("X" if cell.border else str(len(cell.adjacents)) for cell in row))

    def update(self):
        """Runs a single tick of simulation mimicking Minecraft rules and chunks. There's no
        guarantee that any blocks will be frozen in any given tick, or multiple blocks can freeze
        in one tick depending on how many chunks the farm is.
        """
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
        """Guarantees a single water block being frozen while still following adjaceny rules
        """
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

    def _get_eff_yield(self) -> int:
        """The effective yield, the number of ice blocks that can exist

        :return: How many ice blocks that can exist
        """
        eff_yield = self._size * (self._size - 1)
        # For farms greater than 7, another diagonal must be used that shouldn't be count
        if self._size > 7:
            eff_yield -= self._size
            # If odd, avoid counting the center twice when using a cross pattern
            if self._size % 2 == 1:
                eff_yield += 1

        return eff_yield

    def __str__(self):
        """Returns the ice farm as a series of lines that can be used for the terminal

        :return: Returns the ice farm as a string representation
        """
        return "\n".join(" ".join(str(cell) for cell in row) for row in self._grid)

class Screen:
    """A tool to print out a single ice farm that functions as a progress bar.
    """
    def __init__(self, farm_size: int, size_count: int, run_count: int, threshold: float = 1.0):
        """Generates a Screen that prints an ice farm that functions as a progress bar,
        along with relevant metadata and status messages

        :param farm_size: The size of the ice farm, excluding the borders
        :param size_count: How many sizes are being tested
        :param run_count: How many runs per size
        :param threshold: At what yield threshold to stop at from 0.0 to 1.0, defaults to 1.0
        """
        self._farm = IceFarm(farm_size)
        self._size_count = size_count
        self._run_count = run_count
        self._total_runs = self._size_count * self._run_count
        self._yield_threshold = self._farm.eff_yield * threshold

        self._yield = 0
        self._run = 0
        self._status = ""
        self._warning = ""
        self._error = ""
        self._warning_timeout = 0
        self._warning_start_time = 0

    @property
    def status(self) -> str:
        """A status message to print below the metadata

        :return: A string
        """
        return self._status

    @status.setter
    def status(self, value: str):
        """A status message to print below the metadata

        :param value: A string, ideally excluding the newline
        """
        self._status = value

    @property
    def warning(self) -> str:
        """A warning message to print below the status message

        :return: A string
        """
        return self._warning

    @warning.setter
    def warning(self, value: str):
        """A warning message to print below the status message. If a warning timeout is set, this
        value is automatically cleared after self.warning_timeout seconds

        :param value: A string, ideally excluding the newline
        """
        self._warning = value

    @property
    def error(self) -> str:
        """An error message to print below the warning message

        :return: A string
        """
        return self._error

    @error.setter
    def error(self, value: str):
        """An error message to print below the warning message

        :param value: A string, ideally excluding the newline
        """
        self._error = value

    @property
    def warning_timeout(self) -> float:
        """How long to leave the warning displayed for in seconds. If a warning timeout is set
        greater than 0, self.warning is cleared out. Otherwise, warning will print indefinitely

        :return: A float >= 0.0. 0.0 means an indefinite warning
        """
        return self._warning_timeout

    @warning_timeout.setter
    def warning_timeout(self, value: float):
        """How long to leave the warning displayed for in seconds.

        :param value: Any value >= 0.0
        """
        self._warning_timeout = max(value, 0)

    def update(self, current_size: int):
        """Increments the Screen's progress by 1 and refreshes the screen.

        :param current_size: A current size being processed, to be displayed
        in the metatdata
        """
        self._run += 1
        completion_ratio = self._run / self._total_runs
        new_yield = int(self._yield_threshold * completion_ratio)

        if new_yield > self._yield:
            for _ in range(new_yield - self._yield):
                self._farm.increment()
                self.refresh(current_size)

            self._yield = new_yield

    def refresh(self, current_size: int):
        """Clears out the terminal and re-prints the farm, metadata, and status messages
        The order is:
        [Farm]
        [Current Size] [Current Run] [Percentage of Farm filled] [Percentage of Total Progress]
        [Status]
        [Warning]
        [Error]

        :param current_size: A current size being processed, to be displayed
        """
        self._clear()
        print(self._farm, file=sys.stderr)
        print((
                f"{current_size} {(self._run % self._run_count)}"
                f" {self._farm.count / self._farm.eff_yield:.1%}"
                f" {self._run / self._total_runs:.1%}"
            ),
            file=sys.stderr
        )
        if self._status:
            print(self._status, file=sys.stderr)
        if self._warning:
            if self._warning_timeout > 0:
                if self._warning_start_time <= 0:
                    self._warning_start_time = time.time()
                elif time.time() - self._warning_start_time > self._warning_timeout:
                    self._warning = ""
            if self._warning:
                print(self._warning, file=sys.stderr)
        if self._error:
            print(self._error, file=sys.stderr)

    @staticmethod
    def _clear():
        """Clears out the terminal in a platform agnostic way
        """
        os.system("cls" if os.name == "nt" else "clear")


def simulate_generation(size: int, threshold: float = 1.0) -> int:
    """Simulates standard ice farm generation until the threshold * effective yield is met.

    :param size: The size of the ice farm in one-dimension
    :param threshold: The threshold to stop at from 0.0 to 1.0, defaults to 1.0
    :return: The number of ticks it took to reach the threshold
    """
    farm = IceFarm(size)
    eff_yield = farm.eff_yield
    ticks = 0

    while farm.count < eff_yield * threshold:
        ticks += 1
        farm.update()

    return ticks


def simulate_center(size: int) -> float:
    """Simulates the ice farm generation until the center is reached by one of the adjacent ice
    blocks.

    :param size: The size of the ice farm in one-dimension
    :return: The frozen block count / the effective yield
    """
    farm = IceFarm(size)
    ticks = 0

    while farm.count < farm.eff_yield:
        ticks += 1
        farm.update()

        if farm.is_center_reached:
            return farm.count / farm.eff_yield

    return farm.count / farm.eff_yield


def simulate_moments(size: int, threshold: float = 1.0) -> pd.DataFrame:
    """Simulates ice generation to where it tracks every moment new ice is formed in # of ticks

    :param size: The size of the ice farm in one-dimension
    :param threshold: The threshold to stop at from 0.0 to 1.0, defaults to 1.0
    :return: A Pandas DataFrame consisting of farm size ["size"], effective yield ["eff_yield"],
    yield ["yield"], total time (in ticks) ["total_time"], time percentage ["time_ratio"] 
    """
    farm = IceFarm(size)
    ticks = 0

    yields = [0]
    times = [0]

    last_count = farm.count
    while farm.count < farm.eff_yield * threshold:
        ticks += 1
        farm.update()

        if farm.count > last_count:
            last_count = farm.count

            yields.append(farm.count)
            times.append(ticks)

    return pd.DataFrame({
        "size": size,
        "eff_yield": farm.eff_yield,
        "yield": yields,
        "total_time": ticks,
        "time_ratio": [t / ticks for t in times]
    })

CHUNK_SIZE = 16
# Each block has a 1 in 16th chance of being weather updated
WEATHER_UPDATE_CHANCE = 1 / 16
MIN_SIZE = 3

DEFAULT_SIZE = 7
DEFAULT_THRESHOLD = 0.95
DEFAULT_RUN_COUNT = 1000

SCREEN_SIZE = 17
WARNING_TIMEOUT = 7

def main():
    """The main entry point, meant to be called through the terminal
    """
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
            f" {DEFAULT_THRESHOLD} (aka. 95% of the available blocks"
            f" are filled) for standard mode. Defaults to 1.0 for"
            f" --moment mode. This value is ignored when using"
            f" --center."
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

    ignore_threshold = False
    if threshold is None:
        if center_mode or moment_mode:
            threshold = 1.0
        else:
            threshold = DEFAULT_THRESHOLD
    elif center_mode:
        ignore_threshold = True

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("--threshold must be inclusively between 0.0 and 1.0.")

    sizes = list(range(MIN_SIZE if increment else max_size, max_size + 1))
    if moment_mode:
        moments = []

    screen = Screen(SCREEN_SIZE, len(sizes), run_count, threshold)
    if ignore_threshold and center_mode:
        screen.warning = "Warning: --center set, threshold will be ignored"
        screen.warning_timeout = WARNING_TIMEOUT
    screen.refresh(sizes[0])

    if out_path:
        out_path.unlink(True)
        out_path.touch()

    for size in sizes:
        if center_mode:
            with multiprocessing.Pool() as pool:
                runs = []
                for run in pool.imap_unordered(simulate_center, repeat(size, run_count)):
                    runs.append(run)
                    screen.update(size)

            # Print the field size, min, max, average, and median yield percentage
            data = (
                f"{size},{min(runs)},{max(runs)},{sum(runs) / len(runs)}"
                f",{list(sorted(runs))[len(runs) // 2]}"
            )
            screen.status = data
            screen.refresh(size)
            if out_path is not None:
                with out_path.open("a") as fd:
                    fd.write(f"{data}\n")
        elif moment_mode:
            with multiprocessing.Pool() as pool:
                runs = []

                for run in pool.imap_unordered(
                        functools.partial(simulate_moments, size),
                        repeat(threshold, run_count)):
                    runs.append(run)
                    screen.update(size)

                df = pd.concat(runs)
                sub_moments_df = df.groupby(
                    by=["size", "yield"], group_keys=True).mean().reset_index()
                sub_moments_df["yield_ratio"] = (sub_moments_df["yield"] /
                                                 sub_moments_df["eff_yield"])
                moments.append(sub_moments_df)
        else:
            with multiprocessing.Pool() as pool:
                runs = []
                for run in pool.imap_unordered(
                        functools.partial(simulate_generation, size),
                        repeat(threshold, run_count)):
                    runs.append(run)
                    screen.update(size)

            # Print the field size, min, max, average, and median tick runtime
            data = (
                f"{size},{min(runs)},{max(runs)},{sum(runs) / len(runs)}"
                f",{list(sorted(runs))[len(runs) // 2]}"
            )
            screen.status = data
            screen.refresh(size)
            if out_path is not None:
                with out_path.open("a") as fd:
                    fd.write(f"{data}\n")

    if moment_mode:
        moments_df = pd.concat(moments)
        fig = px.line(
            moments_df, color="size", markers=True,
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
        fig.add_hline(y=DEFAULT_THRESHOLD, line_dash="dash",
                      annotation_text=f"{DEFAULT_THRESHOLD:.0%}",
                      annotation_position="bottom right")
        fig.show()

if __name__ == "__main__":
    main()
