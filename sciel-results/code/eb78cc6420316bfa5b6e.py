# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import time

_T0 = time.time()
_DEADLINE = _T0 + 285.0  # Safety margin before 300s limit


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Compute edge-attractiveness matrix for ACO on CVRP.
    
    Combines Delaunay triangulation structure with Clarke-Wright savings
    and demand-aware weighting to guide ants toward non-crossing,
    capacity-feasible routes.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple inverse distance
        dm = np.asarray(distance_matrix, dtype=np.float64)
        d_safe = np.where(dm > 0, dm, 1e-9)
        base = 1.0 / (d_safe + 1e-9)
        np.fill_diagonal(base, 1e-9)
        base[base <= 0] = 1e-9
        return base

    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)

    # Base inverse distance heuristic
    d_safe = np.where(dm > 0, dm, 1e-9)
    base = 1.0 / (d_safe + 1e-9)

    # Build Delaunay triangulation on all points (depot + customers)
    try:
        tri = Delaunay(coords)
    except Exception:
        # Fallback if triangulation fails
        np.fill_diagonal(base, 1e-9)
        base[base <= 0] = 1e-9
        return base

    # Extract edges from triangulation
    delaunay_edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = int(simplex[i])
            b = int(simplex[(i + 1) % 3])
            if a != b:
                delaunay_edges.add((min(a, b), max(a, b)))

    # Add nearest-neighbor edges (k=8) to ensure better connectivity
    try:
        tree = cKDTree(coords)
        _, nn_indices = tree.query(coords, k=9)  # 9 including self
        for i in range(n):
            for j in nn_indices[i, 1:]:
                j = int(j)
                if j != i:
                    delaunay_edges.add((min(i, j), max(i, j)))
    except Exception:
        pass

    # Compute Clarke-Wright savings ratio for customer-customer edges
    d0 = dm[0, :]
    d0_rows = d0[:, np.newaxis]
    d0_cols = d0[np.newaxis, :]
    num = d0_rows + d0_cols - dm
    den = d0_rows + d0_cols
    den_safe = np.where(den > 0, den, 1e-9)
    savings_ratio = np.clip(num / den_safe, 0.0, 1.0)

    # Build the heuristic matrix
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)

    # For edges in the sparse graph, compute a rich weight
    for (a, b) in delaunay_edges:
        dist_ab = dm[a, b]

        # Base: inverse distance (shorter edges are more attractive)
        w = 1.0 / (dist_ab + 1e-9)

        # Clarke-Wright savings bonus for customer-customer edges
        if a > 0 and b > 0:
            sr = savings_ratio[a, b]
            w *= (1.0 + 0.8 * sr)

        # Demand compatibility bonus
        if a == 0 or b == 0:
            # Depot edges: slight bonus for lower-demand customers
            other = b if a == 0 else a
            if other > 0:
                w *= (1.0 + 0.15 * (capacity - dem[other]) / capacity)
        else:
            # Customer-customer: bonus if both fit in one route segment
            if dem[a] + dem[b] <= capacity:
                w *= 1.3
            else:
                w *= 0.7

        heuristic[a, b] = w
        heuristic[b, a] = w

    # For non-Delaunay edges, keep them very low but not zero
    # This enforces the planar structure while allowing some flexibility
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) not in delaunay_edges and (j, i) not in delaunay_edges:
                heuristic[i, j] = 1e-9

    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)

    # Ensure all off-diagonal entries are positive
    mask = heuristic <= 0
    heuristic[mask] = 1e-9

    # Normalize to a reasonable scale
    positive_vals = heuristic[heuristic > 1e-8]
    if len(positive_vals) > 0:
        med = np.median(positive_vals)
        if med > 0:
            heuristic = heuristic / med

    return heuristic
# EVOLVE-BLOCK-END
