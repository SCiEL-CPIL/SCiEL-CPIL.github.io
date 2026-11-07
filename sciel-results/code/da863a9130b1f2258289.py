"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP using a
    cooling-schedule blend of distance-focused and demand-aware heuristics,
    plus a local-density (cluster) term that smooths the demand penalty.
    
    Args:
        distance_matrix: (n, n) symmetric distance matrix
        coordinates: (n, 2) coordinates
        demands: (n,) demands (node 0 is depot with demand 0)
        capacity: vehicle capacity
        
    Returns:
        (n, n) heuristic matrix where larger = more attractive
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s limit

    n = distance_matrix.shape[0]
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)

    # Precompute a few useful vectors
    demand = np.asarray(demands, dtype=float).copy()
    demand[0] = 0.0
    demand_ratio = demand / capacity

    # Local density: for each node, the sum of 1/dist to its k nearest neighbours
    # (excluding self). This is a smooth surrogate for "cluster tightness".
    # Nodes in dense clusters have high density -> edges within clusters are
    # more attractive (they tend to be on the same route).
    k = min(8, n - 1)
    # For each row i, take the k smallest finite distances
    d_work = d.copy()
    np.fill_diagonal(d_work, np.inf)
    # k smallest per row
    k_small = np.argpartition(d_work, k, axis=1)[:, :k]
    density = np.zeros(n)
    for i in range(n):
        idx = k_small[i]
        vals = d_work[i, idx]
        # sum of inverse distances to k nearest
        density[i] = np.sum(1.0 / vals)
    # Normalise density to [0, 1] roughly
    dmax = density.max() if density.max() > 0 else 1.0
    density_norm = density / dmax

    # Cooling-schedule phases: early = high alpha (distance), late = lower alpha
    # blended with a demand-aware term. We compute a weighted sum over phases.
    # Phase t in [0,1]: alpha(t) = 3.0 - 1.0*t  (3.0 early -> 2.0 late)
    # demand_weight(t) = 0.0 + 0.5*t  (0 early -> 0.5 late)
    # cluster_weight(t) = 0.15 * (1 - t)  (stronger early to form clusters)

    nphases = 7
    H = np.zeros((n, n))
    total_weight = 0.0

    for p in range(nphases):
        t = p / max(nphases - 1, 1)
        alpha_p = 3.0 - 1.0 * t
        demand_w = 0.0 + 0.5 * t
        cluster_w = 0.15 * (1.0 - t)

        # Phase weight: use a smooth bump, peak in the middle
        phase_weight = np.exp(-2.0 * (t - 0.5) ** 2)

        # Distance term
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_term = 1.0 / (d ** alpha_p)
        dist_term = np.where(np.isfinite(dist_term), dist_term, 0.0)

        # Demand-aware term: penalise high-demand destinations more when
        # remaining capacity is uncertain (i.e., not from depot).
        # Use a smooth penalty: exp(-demand_w * demand_ratio[j])
        # This is in (0, 1] and smooth.
        demand_penalty = np.exp(-demand_w * demand_ratio)  # (n,)
        # For depot rows, reduce the penalty (fresh load)
        # Scale: depot rows get penalty closer to 1
        demand_penalty_row = np.outer(
            np.where(np.arange(n) == 0, 0.5, 1.0),  # row factor
            demand_penalty
        )
        # Actually we want per-edge: row factor * column penalty
        # Rebuild: row_scale[i] * demand_penalty[j]
        row_scale = np.where(np.arange(n) == 0, 0.5, 1.0)
        demand_penalty_edge = np.outer(row_scale, demand_penalty)  # (n,n)
        # Clamp to [0.3, 1.0]
        demand_penalty_edge = np.clip(demand_penalty_edge, 0.3, 1.0)

        # Cluster term: boost edges between nodes with similar high density
        # Use a smooth kernel: exp(-|density_norm[i] - density_norm[j]| / 0.5)
        # multiplied by min(density_norm[i], density_norm[j]) so only dense
        # clusters get boosted.
        dens_diff = np.abs(density_norm[:, None] - density_norm[None, :])
        cluster_kernel = np.exp(-dens_diff / 0.5)
        cluster_min = np.minimum(density_norm[:, None], density_norm[None, :])
        cluster_term = cluster_kernel * cluster_min  # (n,n) in [0,1]

        # Combine: H_p = dist_term * demand_penalty_edge * (1 + cluster_w * cluster_term)
        H_p = dist_term * demand_penalty_edge * (1.0 + cluster_w * cluster_term)

        # Zero out diagonal
        np.fill_diagonal(H_p, 0.0)
        H_p = np.where(np.isfinite(H_p), H_p, 0.0)

        H += phase_weight * H_p
        total_weight += phase_weight

        # Time guard
        if time.time() > deadline - 10.0:
            break

    if total_weight > 0:
        H /= total_weight
    else:
        # Fallback: simple inverse distance
        with np.errstate(divide='ignore', invalid='ignore'):
            H = 1.0 / (d ** 2.0)
        np.fill_diagonal(H, 0.0)
        H = np.where(np.isfinite(H), H, 0.0)

    # Ensure all off-diagonal entries are positive and finite
    np.fill_diagonal(H, 1e-9)
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)

    return H
# EVOLVE-BLOCK-END
