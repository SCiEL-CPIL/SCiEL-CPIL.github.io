"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Pareto-front distance in (mu, sigma) space.
    
    Identifies non-dominated candidates where low mu (exploitation) and
    high sigma (exploration) are both desirable. Scores are based on
    convex-hull-like distance to this Pareto front, forcing the search
    to pursue diverse regions simultaneously.
    """
    t_start = time.time()
    
    # Ensure proper types and handle numerical issues
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    eps = 1e-12
    
    # For minimization: we want low mu and high sigma (uncertainty)
    # Non-dominated in the sense: no other point has both lower mu AND higher sigma
    
    # Compute Pareto front: points where no other point dominates them
    # A point i dominates point j if mu[i] <= mu[j] and sigma[i] >= sigma[j]
    # (with at least one strict inequality)
    
    # Efficient Pareto front computation
    # Sort by mu ascending, then find points with strictly increasing sigma
    
    # First, handle the case where we need to identify the Pareto front
    # In our 2D case with objectives (minimize mu, maximize sigma):
    # Sort by mu ascending. The Pareto front consists of points where
    # sigma is strictly greater than all previously seen points (with lower mu).
    
    sort_idx = np.argsort(mu)
    mu_sorted = mu[sort_idx]
    sigma_sorted = sigma[sort_idx]
    
    # Find Pareto front in sorted order
    # A point is on the front if its sigma is greater than the max sigma
    # of all points with strictly lower mu
    pareto_mask = np.zeros(n, dtype=bool)
    
    # Track the running max of sigma as we go through sorted mu
    # But we need to be careful: points with same mu should all be considered
    # together. For simplicity, we'll use a slightly different approach.
    
    # Alternative: for each point, check if any other point dominates it
    # This is O(n^2) but n=2000 is manageable
    # However, let's use the efficient sort-based approach
    
    # Sort by mu ascending, and for ties, sort by sigma descending
    # Then the Pareto front is the subsequence where sigma is strictly increasing
    
    # Actually, let's use a cleaner approach:
    # For minimization of mu and maximization of sigma:
    # Point i is non-dominated if there is no j such that mu[j] < mu[i] and sigma[j] > sigma[i]
    # (or mu[j] <= mu[i] and sigma[j] >= sigma[i] with at least one strict)
    
    # Efficient method: sort by mu, then scan for record highs in sigma
    sort_idx = np.argsort(mu, kind='mergesort')  # stable sort
    mu_sorted = mu[sort_idx]
    sigma_sorted = sigma[sort_idx]
    
    # Find indices where sigma achieves a new maximum (strictly greater than all previous)
    # This gives the Pareto front in the sorted order
    running_max = -np.inf
    pareto_in_sorted = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if sigma_sorted[i] > running_max + eps:
            pareto_in_sorted[i] = True
            running_max = sigma_sorted[i]
    
    # Map back to original indices
    pareto_mask = np.zeros(n, dtype=bool)
    pareto_mask[sort_idx] = pareto_in_sorted
    
    # Get the Pareto front points
    pareto_mu = mu[pareto_mask]
    pareto_sigma = sigma[pareto_mask]
    
    # Now compute distance from each point to the Pareto front
    # We want points that are "close" to the front in a meaningful way
    # For points ON the front, they get high scores
    # For points NOT on the front, we compute how far they are from being on the front
    
    # A point is dominated if there exists a Pareto point with lower mu and higher sigma
    # The "distance" can be measured as how much improvement is possible
    
    # For each non-Pareto point, find the closest Pareto point in a normalized space
    # Normalize mu and sigma to [0, 1] for distance computation
    
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min
    sigma_range = sigma_max - sigma_min
    
    if mu_range < eps:
        mu_range = 1.0
    if sigma_range < eps:
        sigma_range = 1.0
    
    # Normalized coordinates
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    pareto_mu_norm = (pareto_mu - mu_min) / mu_range
    pareto_sigma_norm = (pareto_sigma - sigma_min) / sigma_range
    
    # For each point, compute the minimum Euclidean distance to any Pareto point
    # in the normalized (mu_norm, sigma_norm) space
    # But we want to REWARD points that are on or near the front
    
    # Distance to front: for points on the front, distance = 0
    # For points not on the front, distance > 0
    
    # Compute distance from each point to the Pareto front
    # Use a vectorized approach: for each point, find min distance to any Pareto point
    
    n_pareto = len(pareto_mu)
    
    if n_pareto == 0:
        # Fallback: use UCB
        score = -mu + 1.5 * sigma
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        return score
    
    # Compute pairwise distances (n x n_pareto)
    # This could be memory intensive for large n, but n=2000 is fine
    # mu_diff = mu[:, np.newaxis] - pareto_mu[np.newaxis, :]
    # sigma_diff = sigma[:, np.newaxis] - pareto_sigma[np.newaxis, :]
    # dist = np.sqrt(mu_diff**2 + sigma_diff**2)
    # min_dist = np.min(dist, axis=1)
    
    # But we want to use normalized distances
    # Also, we want to weight the distance appropriately
    
    # Actually, let's think about this differently:
    # The Pareto front represents the trade-off between exploitation and exploration
    # Points on the front are ideal candidates
    # We want to score points based on how close they are to the front
    
    # Compute distance to front in normalized space
    # For efficiency, we'll compute in chunks if needed, but n=2000 should be fine
    
    mu_diff = mu[:, np.newaxis] - pareto_mu_norm[np.newaxis, :]
    sigma_diff = sigma_norm[:, np.newaxis] - pareto_sigma_norm[np.newaxis, :]
    dist = np.sqrt(mu_diff**2 + sigma_diff**2)
    min_dist = np.min(dist, axis=1)
    
    # Points on the front have min_dist = 0
    # Points not on the front have min_dist > 0
    
    # We want to convert distance to a score: lower distance = higher score
    # Use an exponential decay or inverse distance
    
    # The maximum possible distance is sqrt(2) in normalized space
    # Scale distance to [0, 1]
    max_dist = np.sqrt(2.0)
    dist_scaled = min_dist / max_dist
    
    # Score: higher for points closer to the front
    # Use: score = 1 / (1 + alpha * dist_scaled)
    # Or: score = exp(-beta * dist_scaled)
    
    # Let's use a combination: points on the front get a base score,
    # and we also consider the actual mu and sigma values for tie-breaking
    
    # Base score from distance to front
    alpha = 3.0
    base_score = 1.0 / (1.0 + alpha * dist_scaled)
    
    # Add a small exploitation bonus: lower mu gets slightly higher score
    # This helps break ties among points equidistant from the front
    mu_min_val = np.min(mu)
    mu_max_val = np.max(mu)
    if mu_max_val - mu_min_val > eps:
        mu_bonus = 0.1 * (1.0 - (mu - mu_min_val) / (mu_max_val - mu_min_val))
    else:
        mu_bonus = np.zeros_like(mu)
    
    # Add a small exploration bonus: higher sigma gets slightly higher score
    sigma_min_val = np.min(sigma)
    sigma_max_val = np.max(sigma)
    if sigma_max_val - sigma_min_val > eps:
        sigma_bonus = 0.1 * (sigma - sigma_min_val) / (sigma_max_val - sigma_min_val)
    else:
        sigma_bonus = np.zeros_like(sigma)
    
    score = base_score + mu_bonus + sigma_bonus
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: check time
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return -mu
    
    return score
# EVOLVE-BLOCK-END
