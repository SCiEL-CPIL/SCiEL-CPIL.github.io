# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy:
    1. Base Attractiveness: Uses a cost-aware ratio prize_j / (d_ij + d_j0)^alpha.
       This accounts for the cost of moving to j AND returning to origin, which is
       critical for feasibility with max_len=3.
    2. Local Prize Density: Incorporates a node-specific factor that rewards
       nodes located in high-density prize regions. This helps the ACO find
       clusters of valuable nodes that can be visited within the short budget.
    3. Combination: The base heuristic is multiplied by a density factor.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Safety check for finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9
    
    # Cost of moving from i to j and then returning to origin 0
    # cost[i, j] = d[i, j] + d[j, 0]
    dist_to_origin = distance_matrix[:, 0]  # shape (n,)
    cost = distance_matrix + dist_to_origin[None, :]  # shape (n, n)
    
    # Ensure cost is positive
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # Ensure prize is non-negative
    prize_safe = np.maximum(prizes, eps)
    prize_col = prize_safe[None, :]  # shape (1, n)
    
    # Base Heuristic: prize_j / cost_ij^alpha
    # Alpha controls locality. Higher alpha = more local.
    # With max_len=3 and unit square, distances are small.
    alpha = 3.0
    H_base = prize_col / (cost_safe ** alpha)
    
    # Calculate Local Prize Density
    # For each node j, calculate the sum of prizes of neighbors weighted by distance.
    # This acts as a "potential field" where high-value clusters are brighter.
    # We use a Gaussian kernel: w(d) = exp(-d^2 / (2 * sigma^2))
    # sigma = 0.25 is a reasonable scale for 50 points in unit square.
    sigma = 0.25
    # distance_matrix is (n, n). d[j, m] is distance between j and m.
    # We want density[j] = sum_m prize[m] * exp(-d[j, m]^2 / (2 * sigma^2))
    # Note: d[j, j] = 0, so node j contributes its own prize.
    
    # Vectorized calculation of density
    # d_sq = distance_matrix ** 2
    # kernel = np.exp(-d_sq / (2 * sigma ** 2))
    # density = kernel @ prizes  # (n, n) @ (n,) -> (n,)
    
    d_sq = distance_matrix ** 2
    kernel = np.exp(-d_sq / (2 * sigma ** 2))
    density = kernel @ prize_safe  # shape (n,)
    
    # Normalize density to [0, 1] to scale the boost appropriately
    max_density = np.max(density)
    if max_density > 0:
        density_norm = density / max_density
    else:
        density_norm = np.zeros_like(density)
    
    # Combine Base and Density
    # H_final[i, j] = H_base[i, j] * (1 + boost_factor * density_norm[j])
    # The boost factor determines how much "clustering" is preferred.
    # A value of 1.0 means a node in a max-density region is twice as attractive
    # as an isolated node with the same base heuristic.
    boost_factor = 1.5
    density_boost = 1.0 + boost_factor * density_norm[None, :]  # shape (1, n)
    
    H = H_base * density_boost
    
    # Check time
    if time.time() > deadline - 1.0:
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Handle non-finite or non-positive values
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Final safety: ensure shape is correct
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9
        
    return H
# EVOLVE-BLOCK-END
