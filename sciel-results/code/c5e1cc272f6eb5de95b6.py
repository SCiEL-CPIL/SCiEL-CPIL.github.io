"""Heuristic matrix for ant colony optimization of capacitated vehicle routing.

Fresh design: the heuristic is a product of three positive structural cues
(distance, depot-centred angular consistency, and capacity/demand fit) rather
than the single inverse-distance term.  Because the ant transition weight is
pheromone[i,j] * heuristic[i,j] and both start at 1, the heuristic is the only
steering signal early on, so it should encode the strongest a-priori route
structure.  All factors are strictly positive, so no non-positive entries occur;
a finite/shape guard is kept as a defensive safety net.
"""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = distance_matrix.shape[0]

    coords = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demacts if False else demands, dtype=float)

    depot = coords[0]

    # 1) Distance cue: shorter edges are more attractive.
    d = distance_matrix + 1e-9
    dist_factor = 1.0 / d

    # 2) Angular consistency around the depot: prefer to keep the angular
    #    order of visited customers (discourages route crossings).
    dx = coords[:, 0] - depot[0]
    dy = coords[:, 1] - depot[1]
    ang = np.arctan2(dy, dx)
    da = ang[None, :] - ang[:, None]
    da = (da + np.pi) % (2.0 * np.pi) - np.pi
    ang_factor = 1.0 + 0.5 * np.cos(da)          # in [0.5, 1.5]

    # 3) Demand / capacity fit: prefer customers that fit the remaining load
    #    well, reducing the chance of a forced early return to the depot.
    rem = capacity - demands[:, None]            # remaining load if i is current
    fit = np.clip(rem / demands[None, :], 0.0, 1.0)
    demand_factor = 0.5 + 0.5 * fit              # in [0.5, 1.5]

    heuristic = dist_factor * ang_factor * demand_factor

    # Defensive: keep strictly positive and finite (should already hold).
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0.0),
                         heuristic, 1e-9)

    if heuristic.shape != distance_matrix.shape:
        raise ValueError("heuristic shape mismatch")
    if not np.all(np.isfinite(heuristic)):
        raise ValueError("heuristic contains non-finite values")

    return heuristic
# EVOLVE-BLOCK-END
