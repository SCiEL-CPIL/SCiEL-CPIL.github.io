"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty with a repulsive force field centered at tour edges.
    
    Uses a continuous penalty that decays with the distance to the nearest tour edge
    in the metric space, creating an exclusion zone that discourages 2-opt from
    accepting swaps that keep the tour close to its previous state.
    """
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    D = edge_distance
    
    # Extract tour edges
    tour_edges = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.append((a, b))
    
    m = len(tour_edges)
    a_arr = np.array([e[0] for e in tour_edges], dtype=int)
    b_arr = np.array([e[1] for e in tour_edges], dtype=int)
    
    # Compute min_closeness[i,j]: minimum over all tour edges of
    #   min(D[i,a] + D[j,b], D[i,b] + D[j,a])
    # This measures how "close" edge (i,j) is to any tour edge.
    
    min_closeness = np.full((n, n), np.inf)
    
    for t in range(m):
        a = a_arr[t]
        b = b_arr[t]
        da = D[:, a]
        db = D[:, b]
        c1 = da[:, None] + db[None, :]
        c2 = db[:, None] + da[None, :]
        c = np.minimum(c1, c2)
        min_closeness = np.minimum(min_closeness, c)
    
    # Normalize edge_n_used
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)
    
    # Tuned parameters for better exploration:
    # beta controls the width of the exclusion zone
    # gamma controls the strength of the penalty
    # We want a stronger penalty on frequently used edges to escape local optima
    
    beta = 8.0
    gamma = 0.8
    
    # Force field: decays exponentially with closeness
    force_field = np.exp(-beta * min_closeness)
    
    # Combine usage and force field
    # Heavily used edges get stronger penalty
    usage_weight = 0.3 + 0.7 * norm_used
    
    penalty_factor = 1.0 + gamma * usage_weight * force_field
    
    result = D * penalty_factor
    
    # Ensure symmetry and finite values
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    
    # Replace any non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=D.max(), neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
