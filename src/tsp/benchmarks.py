"""TSP performance benchmarks across N and algorithms with statistical aggregates."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .algorithms import (
    BKTParams,
    GAParams,
    HCParams,
    NNParams,
    SAParams,
    solve_bkt,
    solve_ga,
    solve_hc,
    solve_nn,
    solve_sa,
)
from .utils import random_matrix


@dataclass
class BenchmarkRow:
    n: int
    algorithm: str
    cost_mean: float
    cost_std: float
    time_mean: float
    time_std: float


def run_tsp_benchmark(
    sizes: list[int] | None = None,
    repeats: int = 3,
    seed: int = 42,
) -> list[BenchmarkRow]:
    sizes = sizes or [5, 10, 15, 20, 30, 50]
    rows: list[BenchmarkRow] = []

    for n in sizes:
        results = {
            "BKT": {"costs": [], "times": []},
            "NN": {"costs": [], "times": []},
            "HC": {"costs": [], "times": []},
            "SA": {"costs": [], "times": []},
            "GA": {"costs": [], "times": []},
        }

        for rep in range(repeats):
            # Generate a new random instance for each repeat
            matrix = random_matrix(n, seed=seed + n * 100 + rep)

            # Limit BKT to small instances to prevent UI lockup/slowdown
            if n <= 10:
                r = solve_bkt(matrix, BKTParams(mode="exhaustiv"))
                results["BKT"]["costs"].append(r.cost)
                results["BKT"]["times"].append(r.elapsed_s)

            # Run heuristics and metaheuristics
            for name, fn, params in [
                ("NN", solve_nn, NNParams(multistart=True)),
                ("HC", solve_hc, HCParams(restarts=3, max_iterations=1000)),
                ("SA", solve_sa, SAParams(max_iterations=1500)),
                ("GA", solve_ga, GAParams(population_size=50, generations=60)),
            ]:
                r = fn(matrix, params)
                results[name]["costs"].append(r.cost)
                results[name]["times"].append(r.elapsed_s)

        for algo, data in results.items():
            if not data["costs"]:
                continue
            costs = data["costs"]
            times = data["times"]
            rows.append(
                BenchmarkRow(
                    n=n,
                    algorithm=algo,
                    cost_mean=float(np.mean(costs)),
                    cost_std=float(np.std(costs)),
                    time_mean=float(np.mean(times)),
                    time_std=float(np.std(times)),
                )
            )

    return rows
