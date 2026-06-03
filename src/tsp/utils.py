"""TSP utilities: distance matrix, cost, file I/O, random instances."""

from __future__ import annotations

import math
import random
from pathlib import Path


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_cities(
    n: int,
    width: float = 800.0,
    height: float = 600.0,
    margin: float = 50.0,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [
        (rng.uniform(margin, width - margin), rng.uniform(margin, height - margin))
        for _ in range(n)
    ]


def matrix_from_coords(cities: list[tuple[float, float]]) -> list[list[float]]:
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(cities[i], cities[j])
            matrix[i][j] = matrix[j][i] = d
    return matrix


def random_matrix(n: int, lo: int = 1, hi: int = 100, seed: int | None = None) -> list[list[float]]:
    rng = random.Random(seed)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(rng.randint(lo, hi))
            matrix[i][j] = matrix[j][i] = d
    return matrix


def load_matrix_file(path: str | Path) -> tuple[int, list[list[float]]]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = int(lines[0])
    matrix = [[float(x) for x in lines[i + 1].split()] for i in range(n)]
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Invalid distance matrix dimensions in file.")
    return n, matrix


def tour_cost(path: list[int], dist_matrix: list[list[float]]) -> float:
    if not path:
        return 0.0
    n = len(path)
    total = sum(dist_matrix[path[i]][path[(i + 1) % n]] for i in range(n))
    return total


def format_tour(path: list[int]) -> str:
    if not path:
        return ""
    return " -> ".join(map(str, path)) + f" -> {path[0]}"


def save_matrix_file(path: str | Path, matrix: list[list[float]]) -> None:
    n = len(matrix)
    lines = [str(n)]
    for row in matrix:
        lines.append(" ".join(f"{x:.4f}" for x in row))
    Path(path).write_text("\n".join(lines), encoding="utf-8")
