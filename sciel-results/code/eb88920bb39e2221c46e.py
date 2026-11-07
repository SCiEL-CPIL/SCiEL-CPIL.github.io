import numpy as np
import time

# Global state for the Pareto front approach
_pareto_state = {
    'front': None,  # list of (mu_val, sigma_val) tuples representing the Pareto front
    'call_count': 0,
    'last_time': None,
    'best_front': None,  # best front found so far
    'front_history': []
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pareto front selection in (mu, sigma) space.
    
    Strategy:
    1. Identify non-dominated candidates in the (mu, sigma) space
       (lower mu is better for exploitation, higher sigma is better for exploration)
    2. Construct a Pareto front from these non-dominated points
    3. Score candidates based on their distance to this front
       - Candidates on or near the front get high scores
       - This forces diverse exploration across the domain
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-8)
    
    n = len(mu)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        # Fallback to simple UCB if we're running out of time
        kappa = 2.0
        score = -mu + kappa * sigma
        score = np.where(np.isfinite(score), score, -np.inf)
        return score
    
    # Normalize mu and sigma for Pareto analysis
    mu_min, mu_max = mu.min(), mu.max()
    sigma_min, sigma_max = sigma.min(), sigma.max()
    
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    
    mu_norm = (mu - mu_min) / mu_range  # Lower is better
    sigma_norm = (sigma - sigma_min) / sigma_range  # Higher is better
    
    # Find non-dominated points in (mu_norm, sigma_norm) space
    # A point is non-dominated if no other point has both lower mu and higher sigma
    is_dominated = np.zeros(n, dtype=bool)
    
    # Efficient non-dominated front extraction
    # Sort by mu_norm (ascending) and keep track of max sigma_norm seen so far
    sorted_idx = np.argsort(mu_norm)
    mu_sorted = mu_norm[sorted_idx]
    sigma_sorted = sigma_norm[sorted_idx]
    
    # For each point, check if it's dominated by any previous point
    # A point i is dominated if there exists j < i (in sorted order) with 
    # mu[j] <= mu[i] and sigma[j] >= sigma[i]
    # Since we're sorted by mu ascending, we just need to check if sigma[i] <= max(sigma[0:i])
    
    max_sigma_so_far = -np.inf
    non_dominated_indices = []
    
    for i in range(n):
        idx = sorted_idx[i]
        if sigma_sorted[i] > max_sigma_so_far:
            non_dominated_indices.append(idx)
            max_sigma_so_far = sigma_sorted[i]
    
    non_dominated_indices = np.array(non_dominated_indices)
    
    # Limit the Pareto front size to avoid too many points
    MAX_FRONT_SIZE = 100
    if len(non_dominated_indices) > MAX_FRONT_SIZE:
        # Keep evenly spaced points along the front
        step = len(non_dominated_indices) // MAX_FRONT_SIZE
        if step > 0:
            non_dominated_indices = non_dominated_indices[::step][:MAX_FRONT_SIZE]
        else:
            non_dominated_indices = non_dominated_indices[:MAX_FRONT_SIZE]
    
    # Compute distance from each candidate to the Pareto front
    # Use a normalized distance that combines both objectives
    
    if len(non_dominated_indices) == 0:
        # Fallback
        kappa = 2.0
        score = -mu + kappa * sigma
        score = np.where(np.isfinite(score), score, -np.inf)
        return score
    
    # Compute distance to the Pareto front
    # For each candidate, find the closest point on the front
    # and compute a combined distance metric
    
    front_mu = mu_norm[non_dominated_indices]
    front_sigma = sigma_norm[non_dominated_indices]
    
    # Efficient computation: for each candidate, compute distance to all front points
    # and take the minimum
    
    # To avoid O(n * front_size) which could be large, we can use a smarter approach
    # For each candidate, the closest front point can be found by:
    # 1. For exploitation: find front point with lowest mu that is >= candidate's mu
    # 2. For exploration: find front point with highest sigma that is <= candidate's sigma
    # But let's use a simpler approach with a limited front size
    
    # Compute squared Euclidean distance in normalized space
    # candidate distance = min over front points of (mu_c - mu_f)^2 + (sigma_c - sigma_f)^2
    
    # To make this efficient, we'll compute distances to a subset of front points
    # and use a weighted combination
    
    # Actually, let's use a different scoring approach:
    # Score = combination of exploitation potential and exploration potential
    # where exploitation is based on how close to the best mu on the front
    # and exploration is based on how close to the best sigma on the front
    
    # Simpler and more robust approach:
    # 1. Compute a "front proximity" score
    # 2. Combine with a base UCB-like score
    
    # Base score: UCB-like
    kappa = 2.5
    base_score = -mu_norm + kappa * sigma_norm
    
    # Front proximity: how close is this point to the Pareto front?
    # For each candidate, compute the minimum distance to any front point
    # in the normalized (mu, sigma) space
    
    # Efficient computation using broadcasting for small front sizes
    if len(front_mu) <= 50:
        # Compute distances to all front points
        # Shape: (n, len(front))
        mu_diff = mu_norm[:, np.newaxis] - front_mu[np.newaxis, :]
        sigma_diff = sigma_norm[:, np.newaxis] - front_sigma[np.newaxis, :]
        dist_sq = mu_diff ** 2 + sigma_diff ** 2
        min_dist = np.sqrt(np.min(dist_sq, axis=1))
    else:
        # Use a subsample of the front
        subsample_idx = np.linspace(0, len(front_mu) - 1, 50, dtype=int)
        front_mu_sub = front_mu[subsample_idx]
        front_sigma_sub = front_sigma[subsample_idx]
        
        mu_diff = mu_norm[:, np.newaxis] - front_mu_sub[np.newaxis, :]
        sigma_diff = sigma_norm[:, np.newaxis] - front_sigma_sub[np.newaxis, :]
        dist_sq = mu_diff ** 2 + sigma_diff ** 2
        min_dist = np.sqrt(np.min(dist_sq, axis=1))
    
    # Score: higher for points close to the front and with good base score
    # Normalize min_dist to [0, 1]
    min_dist_max = min_dist.max() if min_dist.max() > 0 else 1.0
    front_proximity = 1.0 - min_dist / min_dist_max
    
    # Combine: weight both exploitation/exploration balance and front proximity
    # Points on the front get a boost
    score = base_score + 0.5 * front_proximity
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline again
    if time.time() - start_time > DEADLINE:
        # If we're close to the deadline, just return the base score
        score = base_score
        score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
