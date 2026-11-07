import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP using a repulsive force model.
    
    Strategy:
    Instead of adding a fixed or linear penalty, we apply a multiplicative 
    penalty that is inversely proportional to the edge length. This ensures 
    that short, high-value edges (which are likely part of the local optimum) 
    receive a disproportionately large relative penalty, forcing the 2-opt 
    search to explore alternative, longer connections more aggressively.
    
    The penalty magnitude scales with the cumulative usage count (edge_n_used) 
    and is applied multiplicatively to the distance matrix.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # Get edges in the current local optimum tour
    tour = np.array(local_opt_tour)
    if len(tour) < 2:
        return d_out
    
    # Identify edges in the current tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
    # Close the tour
    a, b = tour[-1], tour[0]
    tour_edges[a, b] = True
    tour_edges[b, a] = True
    
    # Repulsive force parameters
    # Base strength of the repulsion
    beta = 0.5
    # Scaling factor for usage-based repulsion
    gamma = 0.2
    
    # Create a mask for non-zero distances to avoid division by zero
    # We only want to penalize edges that have a positive distance
    # Diagonals are 0, so they are naturally excluded if we handle division carefully
    safe_dist = d_out.copy()
    np.fill_diagonal(safe_dist, 1.0)  # Avoid div by 0 on diagonal
    
    # Calculate inverse distance factor
    # 1 / (dist + epsilon) to avoid inf
    eps = 1e-8
    inv_dist_factor = 1.0 / (safe_dist + eps)
    
    # The penalty is multiplicative.
    # We want to increase the distance of edges that are "bad" (used often or in current tour).
    # Penalty factor = 1 + strength * (usage + tour_indicator) * inv_dist_factor
    
    # Combine usage and current tour presence
    # edge_n_used is a count matrix
    # tour_edges is a boolean matrix
    # We treat the current tour as having an "effective usage" of 1 (or more if already used)
    
    # effective_usage = edge_n_used + tour_edges
    # But we must be careful: if an edge is in the tour, it gets an extra kick.
    
    # Let's define the repulsion strength R(i,j) = beta * (1 + gamma * edge_n_used[i,j]) * inv_dist[i,j]
    # Then new_dist[i,j] = dist[i,j] * (1 + R(i,j))
    
    # However, this might be too strong. Let's tune the parameters.
    # For a unit square, average dist is ~0.07. 1/dist is ~14.
    # If beta=0.1, R ~ 1.4. New dist ~ 2.4 * dist.
    # If usage is high, say 5, and gamma=0.1, R ~ 0.1 * (1 + 0.5) * 14 = 2.1. New dist ~ 3.1 * dist.
    
    # Let's use a slightly softer approach to avoid blowing up the matrix.
    # Multiplicative penalty: d_new = d * (1 + penalty_factor)
    
    # penalty_factor = (alpha_base + alpha_usage * edge_n_used) * inv_dist
    # alpha_base: penalty for being in the current local opt (immediate repulsion)
    # alpha_usage: penalty for historical usage
    
    alpha_base = 0.3
    alpha_usage = 0.1
    
    # We only apply the repulsive force to edges in the current tour or those with high usage?
    # The prompt says: "Add the current tour's edges to edge_n_used, call your function... run 2-opt".
    # The "repulsive force" idea suggests penalizing edges that are "attractive" (short) more strongly.
    
    # Let's apply the inverse-distance weighting to the usage penalty primarily, 
    # and add a constant-ish penalty for current tour edges but also weighted by inverse distance.
    
    # Actually, the directive says: "penalty for each edge is inversely proportional to the Euclidean distance"
    # This implies the penalty magnitude P(i,j) = K / d(i,j).
    
    # Let's construct the penalty matrix P
    # P(i,j) = (base_penalty * tour_indicator + usage_penalty * edge_n_used) / (d(i,j) + eps)
    
    # But we must ensure we don't penalize the diagonal.
    # And we should probably only penalize edges that are "candidates" or just all edges?
    # GLS typically penalizes edges in the current local optimum.
    # The "repulsive" idea is to make short edges (which are good) look bad temporarily.
    
    # Let's apply the penalty to ALL edges, but scale it by usage and tour membership?
    # No, usually GLS penalizes the edges in the local optimum.
    # If we penalize ALL edges based on 1/d, we distort the whole metric.
    # The prompt says: "Apply a repulsive force model where the penalty for each edge is inversely proportional..."
    # It doesn't explicitly restrict to tour edges, but context implies penalizing the "attractive" ones.
    # However, edge_n_used is accumulated. 
    # Let's assume the penalty applies to edges that have been used (edge_n_used > 0) or are in the current tour.
    
    # Let's define the penalty only for edges where (edge_n_used > 0 OR tour_edges)
    # This targets the "bad" edges.
    
    penalty_mask = (edge_n_used > 0) | tour_edges
    
    # Initialize penalty matrix
    penalty = np.zeros((n, n))
    
    # Calculate raw penalty weights
    # Weight = (alpha_base * tour_edges + alpha_usage * edge_n_used)
    # Note: tour_edges is boolean, so 1.0 if in tour.
    # edge_n_used is count.
    
    raw_weight = alpha_base * tour_edges.astype(float) + alpha_usage * edge_n_used
    
    # Apply inverse distance scaling
    # Only where penalty_mask is True
    # penalty[i,j] = raw_weight[i,j] / (d[i,j] + eps)
    
    # We must be careful with division by zero for non-diagonal zero distances (shouldn't exist for random points)
    # But just in case, we use safe_dist.
    
    # We only want to apply this to the masked edges.
    # To avoid division issues in the unmasked part, we can compute for all and then mask.
    
    # Compute scaled penalty for all edges
    scaled_penalty = raw_weight / (safe_dist + eps)
    
    # Zero out penalties where we don't want them (diagonal and unmasked)
    # Diagonal is handled by safe_dist=1, but raw_weight is 0 on diagonal (tour_edges false, edge_n_used 0 usually? 
    # Actually edge_n_used might not be 0 on diagonal if previous tours had self-loops? No, TSP tours don't have self-loops.
    # So edge_n_used diagonal is 0. tour_edges diagonal is 0. So raw_weight diagonal is 0.
    # So scaled_penalty diagonal is 0.
    
    # Apply mask
    scaled_penalty[~penalty_mask] = 0.0
    
    # Additive penalty or Multiplicative?
    # "Repulsive force" often implies adding a potential.
    # If we add P to d, then d_new = d + P.
    # P = K / d.
    # If d is small, P is large.
    # If d is large, P is small.
    # This makes short edges very expensive.
    
    # Let's try additive first, as it's more standard for GLS.
    # d_out = d + scaled_penalty
    
    # But wait, if d is very small (e.g. 0.01), 1/d is 100.
    # If alpha_base is 0.3, penalty is 30.
    # d_new = 0.01 + 30 = 30.01.
    # This is a huge change. It effectively removes the edge.
    # Is this desirable? Yes, to escape local optima, we want to force the solver to pick different edges.
    
    # However, if we do this for ALL edges in the tour, we might destroy the tour structure too much.
    # But 2-opt will rebuild it.
    
    # Let's refine the parameters to be less extreme.
    # Instead of 1/d, maybe 1/(d + d_min) or similar?
    # Or just lower the alpha.
    
    # Let's try alpha_base = 0.05, alpha_usage = 0.01
    # If d=0.07, 1/d=14.
    # Penalty = 0.05 * 14 = 0.7.
    # d_new = 0.07 + 0.7 = 0.77.
    # This is a 10x increase. Significant but not destroying.
    
    # Let's stick with these lower alphas.
    
    alpha_base = 0.05
    alpha_usage = 0.01
    
    # Recalculate raw_weight with new alphas
    raw_weight = alpha_base * tour_edges.astype(float) + alpha_usage * edge_n_used
    
    # Recalculate scaled penalty
    scaled_penalty = raw_weight / (safe_dist + eps)
    scaled_penalty[~penalty_mask] = 0.0
    
    # Add to distance matrix
    d_out = d_out + scaled_penalty
    
    # Ensure diagonal stays 0
    np.fill_diagonal(d_out, 0.0)
    
    # Ensure symmetry
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure finite values
    d_out = np.nan_to_num(d_out, nan=0.0, posinf=1e10, neginf=1e10)
    
    # Ensure non-negative distances
    d_out = np.maximum(d_out, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        return d_out
    
    return d_out
