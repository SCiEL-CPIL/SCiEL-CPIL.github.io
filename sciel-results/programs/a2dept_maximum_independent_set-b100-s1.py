import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s / 150s kill

    n = adjacency.shape[0]
    A = adjacency.astype(float)

    # 1. Degree
    deg = A.sum(axis=1)

    # 2. Mean Neighbor Degree
    sum_neighbor_deg = A @ deg
    mean_neighbor_deg = np.zeros(n, dtype=float)
    mask = deg > 0
    mean_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]

    # 3. Sum of Inverse Neighbor Degrees (lower = neighbors are hubs)
    inv_deg = 1.0 / (deg + 1.0)
    sum_inv_neighbor_deg = A @ inv_deg

    # 4. Exact Triangle Count via diag(A @ A)
    A2 = A @ A
    tri_counts = np.diag(A2) / 2.0

    # 5. Neighbor-of-Hub: sum of neighbor degrees (raw)
    #    (redundant with mean_nbr*deg but captures hub-adjacency directly)

    def zscore(x):
        x = x.astype(float)
        mean = x.mean()
        std = x.std()
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std

    z_deg = zscore(deg)
    z_mean_nbr = zscore(mean_neighbor_deg)
    z_sum_inv = zscore(sum_inv_neighbor_deg)
    z_tri = zscore(tri_counts)

    # Weights (tuned within family)
    # -deg: low degree good
    # +mean_nbr: high neighbor degree good (block hubs)
    # -sum_inv: low sum of inverse (neighbors are high-degree) good
    # -tri: low triangle (sparse, less local redundancy) good
    w_deg = 1.0
    w_mean_nbr = 0.5
    w_sum_inv = 0.5
    w_tri = 0.2

    scores = (-w_deg * z_deg
              + w_mean_nbr * z_mean_nbr
              - w_sum_inv * z_sum_inv
              - w_tri * z_tri)

    # Isolated vertices: always safe to pick, give huge priority
    scores[~mask] = 1e6

    if time.time() > deadline:
        return scores

    return scores
