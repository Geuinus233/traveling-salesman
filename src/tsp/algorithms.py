"""TSP solvers: BKT, NN, HC, SA, GA — parametrizable."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from .utils import tour_cost


@dataclass
class TSPResult:
    path: list[int]
    cost: float
    elapsed_s: float
    meta: dict = field(default_factory=dict)


@dataclass
class BKTParams:
    mode: str = "exhaustiv"  # exhaustiv | prima | y_solutii | timp
    y_solutions: int = 5
    time_limit_s: float = 30.0


@dataclass
class NNParams:
    start: int = 0
    multistart: bool = False


@dataclass
class HCParams:
    max_iterations: int = 5000
    restarts: int = 10
    seed: int | None = None


@dataclass
class SAParams:
    initial_temp: float = 1000.0
    cooling_rate: float = 0.995
    min_temp: float = 0.01
    max_iterations: int = 5000
    seed: int | None = None


@dataclass
class GAParams:
    population_size: int = 80
    generations: int = 150
    mutation_rate: float = 0.15
    crossover_rate: float = 0.85
    elitism: int = 2
    seed: int | None = None


def _two_opt_neighbor(path: list[int], rng: random.Random) -> list[int]:
    i, j = sorted(rng.sample(range(len(path)), 2))
    if j - i < 2:
        return path[:]
    new_path = path[:]
    new_path[i : j + 1] = reversed(new_path[i : j + 1])
    return new_path


def solve_bkt(dist_matrix: list[list[float]], params: BKTParams | None = None) -> TSPResult:
    params = params or BKTParams()
    n = len(dist_matrix)
    start = time.perf_counter()

    best_cost = math.inf
    best_path: list[int] = []
    solutions_found = 0
    deadline = start + params.time_limit_s

    def backtrack(path: list[int], visited: list[bool], cost: float) -> bool:
        nonlocal best_cost, best_path, solutions_found

        if params.mode == "timp" and time.perf_counter() >= deadline:
            return True

        if len(path) == n:
            total = cost + dist_matrix[path[-1]][path[0]]
            solutions_found += 1
            if total < best_cost:
                best_cost = total
                best_path = path[:]
            if params.mode == "prima":
                return True
            if params.mode == "y_solutii" and solutions_found >= params.y_solutions:
                return True
            return False

        current = path[-1]
        for nxt in range(n):
            if visited[nxt]:
                continue
            new_cost = cost + dist_matrix[current][nxt]
            if new_cost >= best_cost:
                continue
            visited[nxt] = True
            path.append(nxt)
            if backtrack(path, visited, new_cost):
                return True
            path.pop()
            visited[nxt] = False
        return False

    visited = [False] * n
    visited[0] = True
    backtrack([0], visited, 0.0)
    elapsed = time.perf_counter() - start
    return TSPResult(
        path=best_path,
        cost=best_cost if best_path else math.inf,
        elapsed_s=elapsed,
        meta={"solutions_found": solutions_found, "mode": params.mode},
    )


def solve_nn(dist_matrix: list[list[float]], params: NNParams | None = None) -> TSPResult:
    params = params or NNParams()
    n = len(dist_matrix)
    start = time.perf_counter()

    def nn_from(s0: int) -> tuple[list[int], float]:
        unvisited = set(range(n)) - {s0}
        path = [s0]
        current = s0
        while unvisited:
            nxt = min(unvisited, key=lambda c: dist_matrix[current][c])
            unvisited.remove(nxt)
            path.append(nxt)
            current = nxt
        return path, tour_cost(path, dist_matrix)

    if params.multistart:
        best_path, best_cost = [], math.inf
        for s in range(n):
            p, c = nn_from(s)
            if c < best_cost:
                best_path, best_cost = p, c
    else:
        best_path, best_cost = nn_from(params.start % n)

    return TSPResult(
        path=best_path,
        cost=best_cost,
        elapsed_s=time.perf_counter() - start,
        meta={"multistart": params.multistart},
    )


def solve_hc(dist_matrix: list[list[float]], params: HCParams | None = None) -> TSPResult:
    params = params or HCParams()
    rng = random.Random(params.seed)
    n = len(dist_matrix)
    start = time.perf_counter()

    global_best_path: list[int] = list(range(n))
    global_best_cost = math.inf
    history: list[float] = []

    for _ in range(max(1, params.restarts)):
        path = list(range(n))
        rng.shuffle(path)
        current = path[:]
        current_cost = tour_cost(current, dist_matrix)
        
        if current_cost < global_best_cost:
            global_best_cost = current_cost
            global_best_path = current[:]
        history.append(global_best_cost)

        improved = True
        step_count = 0
        while improved and step_count < (params.max_iterations // params.restarts):
            improved = False
            best_neighbor, best_cost = current, current_cost
            for i in range(n - 1):
                for j in range(i + 2, n if i > 0 else n - 1):
                    neighbor = current[:]
                    neighbor[i : j + 1] = reversed(neighbor[i : j + 1])
                    c = tour_cost(neighbor, dist_matrix)
                    if c < best_cost:
                        best_neighbor, best_cost = neighbor, c
                        improved = True
            current, current_cost = best_neighbor, best_cost
            if current_cost < global_best_cost:
                global_best_cost = current_cost
                global_best_path = current[:]
            step_count += 1
            history.append(global_best_cost)

    return TSPResult(
        path=global_best_path,
        cost=global_best_cost,
        elapsed_s=time.perf_counter() - start,
        meta={"restarts": params.restarts, "history": history},
    )


def solve_sa(dist_matrix: list[list[float]], params: SAParams | None = None) -> TSPResult:
    params = params or SAParams()
    rng = random.Random(params.seed)
    n = len(dist_matrix)
    start = time.perf_counter()

    current = list(range(n))
    rng.shuffle(current)
    current_cost = tour_cost(current, dist_matrix)
    best, best_cost = current[:], current_cost
    temp = params.initial_temp
    history: list[float] = []

    for idx in range(params.max_iterations):
        if temp < params.min_temp:
            break
        neighbor = _two_opt_neighbor(current, rng)
        neighbor_cost = tour_cost(neighbor, dist_matrix)
        delta = neighbor_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current, current_cost = neighbor, neighbor_cost
            if current_cost < best_cost:
                best, best_cost = current[:], current_cost
        temp *= params.cooling_rate
        if idx % max(1, params.max_iterations // 500) == 0:
            history.append(best_cost)

    return TSPResult(
        path=best,
        cost=best_cost,
        elapsed_s=time.perf_counter() - start,
        meta={"final_temp": temp, "history": history},
    )


def _order_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[a : b + 1] = p1[a : b + 1]
    fill = [g for g in p2 if g not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child


def _mutate_swap(path: list[int], rng: random.Random) -> None:
    i, j = rng.sample(range(len(path)), 2)
    path[i], path[j] = path[j], path[i]


def solve_ga(dist_matrix: list[list[float]], params: GAParams | None = None) -> TSPResult:
    params = params or GAParams()
    rng = random.Random(params.seed)
    n = len(dist_matrix)
    start = time.perf_counter()

    def fitness(path: list[int]) -> float:
        return -tour_cost(path, dist_matrix)

    def random_individual() -> list[int]:
        ind = list(range(n))
        rng.shuffle(ind)
        return ind

    population = [random_individual() for _ in range(params.population_size)]
    best = max(population, key=fitness)
    history: list[float] = []

    for _ in range(params.generations):
        population.sort(key=fitness, reverse=True)
        gen_best = population[0]
        if fitness(gen_best) > fitness(best):
            best = gen_best[:]
        history.append(-fitness(gen_best))

        next_pop = [p[:] for p in population[: params.elitism]]
        while len(next_pop) < params.population_size:
            p1, p2 = rng.sample(population[: max(4, params.population_size // 2)], 2)
            child = p1[:] if rng.random() > params.crossover_rate else _order_crossover(p1, p2, rng)
            if rng.random() < params.mutation_rate:
                _mutate_swap(child, rng)
            next_pop.append(child)
        population = next_pop

    return TSPResult(
        path=best,
        cost=tour_cost(best, dist_matrix),
        elapsed_s=time.perf_counter() - start,
        meta={"generations": params.generations, "history": history},
    )


SOLVERS: dict[str, Callable[..., TSPResult]] = {
    "BKT": solve_bkt,
    "NN": solve_nn,
    "HC": solve_hc,
    "SA": solve_sa,
    "GA": solve_ga,
}
