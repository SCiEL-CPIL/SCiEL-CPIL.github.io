"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for the ant colony.
    
    The heuristic encourages moves to nodes j that:
    1. Have high prize value.
    2. Are part of high-value clusters (neighborhood density).
    3. Have low incremental cost (d[i,j]) and low return cost (d[j,0]).
    4. Leave sufficient budget slack (penalize high total commitment).
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
        
    # Ensure inputs are clean
    prizes = np.maximum(prizes, 0.0)
    distance_matrix = np.maximum(distance_matrix, 0.0)
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    
    eps = 1e-9
    
    # 1. Compute Neighborhood Density
    # For each node j, how much prize is available in its neighborhood?
    # density[j] = sum_k (prize[k] / (d[j,k] + 1.0))
    # Using +1.0 in denominator prevents explosion for very close nodes and acts as a smooth kernel.
    # We exclude self (k=j) implicitly or explicitly? prize[j] is handled separately.
    # d[j,j] is 0, so prize[j]/1.0 would be added. We should probably exclude self to avoid double counting 
    # or just accept it as a self-bonus. Let's exclude self to be precise about "neighbors".
    
    # Create a matrix of distances with diagonal set to inf to exclude self in sum
    dist_for_density = distance_matrix.copy()
    np.fill_diagonal(dist_for_density, np.inf)
    
    # density_matrix[j, k] = prize[k] / (d[j,k] + 1.0)
    # We want sum over k for each j.
    # Shape: (n, n). prizes shape (n,).
    # density_term = (1 / (dist_for_density + 1.0)) * prizes[None, :]
    # density[j] = sum(density_term[j, :])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / (dist_for_density + 1.0)
    
    inv_dist = np.where(np.isfinite(inv_dist), inv_dist, 0.0)
    
    density_terms = inv_dist * prizes[None, :]
    neighborhood_density = np.sum(density_terms, axis=1) # shape (n,)
    
    # Normalize density to be roughly in [0, 1] or similar scale to not dominate prize
    # Max possible density is sum of all prizes / 1.0 (if all dist=0, but they aren't).
    # Let's just scale it by 1/max_prize or similar? 
    # Actually, if prizes are ~1, density can be up to ~n.
    # Let's use a factor: 1 + neighborhood_density
    # This boosts nodes with many valuable neighbors.
    
    # 2. Compute Cost Matrix
    # Cost of moving i -> j and then being committed to return from j.
    # The "sunk cost" of this decision is d[i,j] (move) + d[j,0] (return).
    # If this sum is > max_len, the move is infeasible from any state (since len_so_far >= 0).
    
    d_ij = distance_matrix
    d_j0 = distance_matrix[:, 0] # shape (n,)
    
    # total_commitment[i, j] = d[i,j] + d[j,0]
    total_commitment = d_ij + d_j0[None, :] # shape (n, n)
    
    # 3. Compute Heuristic
    # Base attractiveness: Prize / Cost
    # Boost: (1 + Density)
    # Penalty: If commitment is high, reduce attractiveness further?
    # The division by Cost already does this.
    # However, if commitment is close to max_len, the ant has no slack for future nodes.
    # We can add a slack factor: max_len / (commitment + eps) ?
    # This would make eta ~ Prize * Density * max_len / (Cost^2).
    # This strongly penalizes long commitments.
    
    # Let's try:
    # numerator = prizes[j] * (1 + neighborhood_density[j])
    # denominator = total_commitment[i,j] + eps
    
    # Additionally, mask out infeasible edges.
    # If total_commitment[i,j] > max_len, set heuristic to 1e-9.
    
    numerator = prizes[None, :] * (1.0 + neighborhood_density[None, :])
    
    # Avoid division by zero
    safe_denom = total_commitment + eps
    
    heuristic = numerator / safe_denom
    
    # Apply infeasibility mask
    # If the move i->j plus return j->0 exceeds max_len, it can never be part of a valid path
    # (since current_length >= 0).
    infeasible_mask = total_commitment > max_len
    heuristic[infeasible_mask] = 1e-9
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
