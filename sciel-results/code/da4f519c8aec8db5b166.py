"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    # Keep the same family: inverse-distance-like edge attractiveness.
    # Use a slightly stronger inverse power to emphasize short edges,
    # then normalize so the heuristic scale is comparable to initial pheromone.
    d = np.maximum(dm, 1e-9)
    h = 1.0 / (d ** 1.25)

    # Normalize to mean 1.0 to avoid overwhelming pheromone at start.
    mean_h = np.mean(h)
    if not np.isfinite(mean_h) or mean_h <= 0.0:
        h = np.ones_like(dm, dtype=float)
    else:
        h = h / mean_h

    # Contract requires positive entries; non-positive becomes 1e-9.
    h = np.where(np.isfinite(h) & (h > 0.0), h, 1e-9)
    return h
# EVOLVE-BLOCK-END
