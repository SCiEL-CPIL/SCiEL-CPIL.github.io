"""Priority scoring for the fixed greedy Maximum Independent Set decoder.

The decoder picks vertices in descending score order, skipping neighbours of
already-picked vertices.  We want a score that makes the greedy pass select a
large independent set.

Design (from scratch):
  * Low degree is good: a low-degree vertex is cheap to include (it blocks few
    other candidates).
  * High mean-neighbour-degree is good: including such a vertex "absorbs" its
    high-degree neighbours, preventing those hubs from later blocking many
    low-degree vertices (the measured hub-blocking benefit).

We combine both into a single per-vertex scalar:
    score = mean_neighbour_degree - degree
which is a cheap vectorized proxy for "independent-set value per blocked
vertex".  The computation is a single O(n) pass, so it is far below the time
budget for 200-node graphs.
"""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Work in float for stable arithmetic; adjacency is a bool/0-1 matrix.
    A = np.asarray(adjacency, dtype=np.float64)

    # Degree of every vertex (number of neighbours).
    degree = A.sum(axis=1)

    # Mean degree over each vertex's neighbours (0.0 for isolated vertices).
    # Sum of neighbour degrees, then normalise by degree (avoid div-by-zero).
    neigh_deg_sum = A @ degree          # for each v: sum of deg(u) over u~v
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_neigh_deg = np.where(degree > 0, neigh_deg_sum / np.maximum(degree, 1.0), 0.0)

    # Hub-blocking benefit minus inclusion cost.  Higher score is selected first.
    score = mean_neigh_deg - degree

    return score
# EVOLVE-BLOCK-END
