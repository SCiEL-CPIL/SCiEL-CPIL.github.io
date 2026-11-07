"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function for Bayesian optimization.
    
    This approach scores candidates by the volume of the Pareto front they
    would add to the set of non-dominated (mu, sigma) points, thereby
    explicitly optimizing for a diverse set of trade-offs rather than a
    single scalarized compromise.
    
    For minimization, we consider the (mu, sigma) space where lower mu is
    better and lower sigma is better. We compute the hypervolume contribution
    of each candidate to the Pareto front of all candidates.
    """
    n = len(mu)
    
    # Normalize sigma to be in a reasonable range for hypervolume computation
    # Use a reference point that is "bad" in both dimensions
    # For minimization, the reference point should be higher than all values
    mu_max = np.max(mu) + 1e-6
    sigma_max = np.max(sigma) + 1e-6
    
    # Reference point for hypervolume (upper bound in both dimensions for minimization)
    ref_mu = mu_max
    ref_sigma = sigma_max
    
    # Find non-dominated points
    # A point i dominates point j if mu[i] <= mu[j] and sigma[i] <= sigma[j]
    # and at least one is strict
    
    # Sort by mu ascending, then by sigma ascending
    sorted_idx = np.lexsort((sigma, mu))
    sorted_mu = mu[sorted_idx]
    sorted_sigma = sigma[sorted_idx]
    
    # Find Pareto front: points that are not dominated by any other point
    # In 2D, after sorting by mu ascending, a point is on the Pareto front
    # if its sigma is lower than all previous points' sigma
    pareto_mask = np.zeros(n, dtype=bool)
    
    # Track the minimum sigma seen so far (from the left, i.e., lower mu)
    min_sigma_so_far = np.inf
    for i in range(n):
        if sorted_sigma[i] < min_sigma_so_far:
            pareto_mask[sorted_idx[i]] = True
            min_sigma_so_far = sorted_sigma[i]
    
    # Compute hypervolume contribution for each point
    # For 2D minimization, the hypervolume contribution of a Pareto point
    # is the area of the rectangle it "owns"
    
    scores = np.zeros(n)
    
    # Get the Pareto points sorted by mu
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) == 0:
        # Fallback: use UCB
        beta = 2.0
        return mu + beta * sigma
    
    pareto_mu = mu[pareto_indices]
    pareto_sigma = sigma[pareto_indices]
    
    # Sort Pareto points by mu ascending
    pareto_order = np.argsort(pareto_mu)
    pareto_mu_sorted = pareto_mu[pareto_order]
    pareto_sigma_sorted = pareto_sigma[pareto_order]
    
    # The Pareto front should have decreasing sigma as mu increases
    # (otherwise some points would be dominated)
    
    # Compute hypervolume contribution for each Pareto point
    # In 2D, for a point (mu_i, sigma_i) on the Pareto front sorted by mu ascending:
    # The width in mu direction is (mu_{i+1} - mu_i) where mu_{n} = ref_mu
    # The height in sigma direction is (sigma_i - sigma_{i-1}) where sigma_{-1} = ref_sigma... 
    # Actually, for minimization with reference point (ref_mu, ref_sigma):
    # The contribution of point i is:
    # (mu_{i+1} - mu_i) * (sigma_i - min(sigma_{j} for j > i))
    # But since it's a Pareto front, sigma is decreasing, so:
    # contribution_i = (mu_{i+1} - mu_i) * (sigma_i - sigma_{i+1}) for i < n-1
    # contribution_{n-1} = (ref_mu - mu_{n-1}) * (sigma_{n-1} - 0) ... 
    # 
    # Actually, let me think more carefully. The hypervolume of a set of points
    # in 2D minimization with reference point (r1, r2) is the area of the region
    # dominated by at least one point and bounded by the reference point.
    #
    # For a Pareto front sorted by mu ascending: (mu_0, sigma_0), (mu_1, sigma_1), ..., (mu_{k-1}, sigma_{k-1})
    # where mu_0 < mu_1 < ... < mu_{k-1} and sigma_0 > sigma_1 > ... > sigma_{k-1}
    #
    # The total hypervolume is the sum of rectangles:
    # For i = 0 to k-2: width = mu_{i+1} - mu_i, height = sigma_i
    # For i = k-1: width = ref_mu - mu_{k-1}, height = sigma_{k-1}
    # Wait, that's not quite right either. Let me reconsider.
    #
    # The region dominated by point i is [mu_i, ref_mu] x [sigma_i, ref_sigma]
    # But we want the union of all such regions. For a Pareto front, this forms
    # a staircase shape. The hypervolume is:
    # Sum over i of (mu_{i+1} - mu_i) * (sigma_i - sigma_{i+1}) + (ref_mu - mu_{k-1}) * sigma_{k-1}
    # where mu_k = ref_mu and sigma_k = 0 (effectively)
    
    k = len(pareto_mu_sorted)
    hypervol_contrib = np.zeros(k)
    
    for i in range(k):
        if i < k - 1:
            width = pareto_mu_sorted[i + 1] - pareto_mu_sorted[i]
            height = pareto_sigma_sorted[i] - pareto_sigma_sorted[i + 1]
        else:
            width = ref_mu - pareto_mu_sorted[i]
            height = pareto_sigma_sorted[i]
        hypervol_contrib[i] = width * height
    
    # Assign scores: Pareto points get their hypervolume contribution,
    # non-Pareto points get 0 (or a small value)
    for i, idx in enumerate(pareto_order):
        original_idx = pareto_indices[idx]
        scores[original_idx] = hypervol_contrib[i]
    
    # For non-Pareto points, we can assign a score based on how close they are
    # to the Pareto front, or just give them 0
    # To make the acquisition more meaningful, let's give non-Pareto points
    # a score proportional to their "distance" to the Pareto front
    
    # Actually, for the acquisition function, we want to select the candidate
    # that would add the most hypervolume if it were the new best point.
    # This is different from the hypervolume contribution in the current front.
    
    # Let me reconsider: the hypervolume-based acquisition typically evaluates
    # the marginal hypervolume contribution of each candidate point to the
    # set of already-evaluated points (or the Pareto front of candidates).
    
    # Since we don't have the evaluated points here, we'll use the Pareto front
    # of the candidates themselves and assign scores based on their contribution.
    
    # Add a small penalty for high mu to prefer lower expected values
    # while still encouraging exploration via the hypervolume term
    scores = scores - 0.1 * mu
    
    return scores
# EVOLVE-BLOCK-END
