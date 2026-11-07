import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Pareto-front selection with convex-hull distance scoring.
    
    Identifies non-dominated candidates in the (mu, sigma) space (lower mu is better,
    higher sigma is better for exploration) to form a diverse Pareto front.
    Scores are based on the convex hull distance to this front, which encourages
    the search to simultaneously pursue low-mean exploitation and high-variance
    exploration across distinct regions of the domain.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    n = mu.shape[0]
    
    # --- Step 1: Identify non-dominated candidates in (mu, sigma) space ---
    # Lower mu is better (exploitation), higher sigma is better (exploration)
    # A point is non-dominated if no other point has both lower mu and higher sigma
    
    # Normalize mu and sigma to [0, 1] for distance calculations
    mu_min, mu_max = mu.min(), mu.max()
    sigma_min, sigma_max = sigma_safe.min(), sigma_safe.max()
    
    mu_range = mu_max - mu_min
    sigma_range = sigma_max - sigma_min
    
    if mu_range < 1e-12:
        mu_range = 1e-12
    if sigma_range < 1e-12:
        sigma_range = 1e-12
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Find non-dominated points: lower mu_norm and higher sigma_norm are better
    # Point i is dominated by point j if mu_j <= mu_i and sigma_j >= sigma_i and (strict in at least one)
    
    # Efficient approach: sort by mu_norm ascending, then find points with increasing sigma_norm
    order = np.argsort(mu_norm)
    sorted_mu = mu_norm[order]
    sorted_sigma = sigma_norm[order]
    
    # A point is on the Pareto front if its sigma_norm is strictly greater than 
    # all previous points with lower or equal mu_norm
    # Process in order of increasing mu_norm
    pareto_mask = np.zeros(n, dtype=bool)
    max_sigma_so_far = -1
    
    for idx in order:
        if sorted_sigma[list(order).index(idx)] > max_sigma_so_far:
            pareto_mask[idx] = True
            max_sigma_so_far = sorted_sigma[list(order).index(idx)]
    
    # More efficient Pareto front identification
    pareto_mask = np.zeros(n, dtype=bool)
    order = np.argsort(mu_norm)
    sorted_sigma = sigma_norm[order]
    
    # Track maximum sigma seen so far as we go through increasing mu
    max_sigma_so_far = -1
    for i in range(n):
        idx = order[i]
        if sorted_sigma[i] > max_sigma_so_far:
            pareto_mask[idx] = True
            max_sigma_so_far = sorted_sigma[i]
    
    # Get the Pareto front points
    pareto_indices = np.where(pareto_mask)[0]
    
    # If no Pareto points found (shouldn't happen), fallback to all points
    if len(pareto_indices) == 0:
        pareto_indices = np.arange(n)
    
    # --- Step 2: Compute convex hull distance to the Pareto front ---
    # We want to score candidates based on their distance to the Pareto front
    # in the (mu_norm, sigma_norm) space. Points ON or NEAR the front get higher scores.
    
    # For each candidate, compute the minimum Euclidean distance to any Pareto front point
    # But we want points that are "above" the front (higher sigma for given mu) to score high
    
    # Alternative: use the convex hull of the Pareto front and compute distance
    # Points inside the convex hull (below the front) get lower scores
    # Points on or above the front get higher scores
    
    # Simpler approach: for each candidate, find the Pareto point with the closest mu_norm
    # and compare sigma_norm. If candidate's sigma_norm >= Pareto point's sigma_norm, it's good.
    
    # Actually, let's use a different structural approach:
    # Score = combination of:
    # 1. Exploitation term: based on how much lower mu is compared to f_best
    # 2. Exploration term: based on sigma
    # 3. Pareto dominance bonus: bonus for being on or near the Pareto front
    
    # Compute a "Pareto rank" - number of points that dominate this point
    # Points with rank 0 are on the Pareto front
    
    # Efficient Pareto rank computation
    # For each point, count how many points have both lower mu and higher sigma
    mu_expanded = mu_norm[:, np.newaxis]
    sigma_expanded = sigma_norm[:, np.newaxis]
    
    # Point j dominates point i if mu_j <= mu_i and sigma_j >= sigma_i and (mu_j < mu_i or sigma_j > sigma_i)
    # This is O(n^2) which is 4 million for n=2000, acceptable
    
    # Actually, let's use a simpler and faster approach:
    # Sort by mu, then for each point, the number of dominating points can be computed
    
    # Let's just compute a score based on the Pareto front distance directly
    # Use the Pareto front points to create a "front curve" and measure distance
    
    # Get Pareto front coordinates
    pf_mu = mu_norm[pareto_indices]
    pf_sigma = sigma_norm[pareto_indices]
    
    # Sort Pareto front by mu
    pf_order = np.argsort(pf_mu)
    pf_mu = pf_mu[pf_order]
    pf_sigma = pf_sigma[pf_order]
    
    # For each candidate, find the closest Pareto point in mu space
    # and compute the "height" difference in sigma space
    
    # Interpolate the Pareto front to get a reference sigma for each candidate's mu
    # Candidates with mu outside the Pareto front range get extrapolated values
    
    # Create a reference curve from the Pareto front
    # For mu values below the minimum Pareto mu, use the highest sigma
    # For mu values above the maximum Pareto mu, use the lowest sigma
    
    pf_mu_min = pf_mu[0]
    pf_mu_max = pf_mu[-1]
    pf_sigma_at_min = pf_sigma[0]  # highest sigma at lowest mu
    pf_sigma_at_max = pf_sigma[-1]  # lowest sigma at highest mu
    
    # Interpolate
    reference_sigma = np.interp(mu_norm, pf_mu, pf_sigma)
    
    # The "excess" sigma is how much above the Pareto front a candidate is
    # Higher excess sigma means the candidate is in a more exploratory region
    excess_sigma = sigma_norm - reference_sigma
    
    # --- Step 3: Construct final score ---
    # Combine exploitation and exploration with Pareto-aware weighting
    
    # Exploitation term: expected improvement-like
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Exploration term: scaled sigma
    sigma_scale = sigma_safe / max(sigma_median := np.median(sigma_safe), 1e-9)
    
    # Pareto bonus: reward being on or above the Pareto front
    # Points with positive excess_sigma are above the front (more exploratory)
    # Points with negative excess_sigma are below the front (dominated)
    pareto_bonus = np.where(excess_sigma >= 0, 1.0, np.exp(-np.abs(excess_sigma) * 5))
    
    # Combine: weighted sum with Pareto-aware modulation
    # When a point is on the Pareto front, trust both exploitation and exploration equally
    # When dominated, downweight the dominated dimension
    
    # Final score: EI + exploration term, modulated by Pareto position
    exploitation_weight = 1.0
    exploration_weight = 0.5
    
    base_score = exploitation_weight * ei + exploration_weight * sigma_scale * 0.1
    
    # Add a bonus for being on the Pareto front
    front_bonus = 0.2 * pareto_bonus
    
    scores = base_score + front_bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
