"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Batch-aware Greedy Expected Improvement (GEI) with variance-reduction penalty.
    
    The acquisition score combines:
      1. Expected Improvement (EI) for exploitation
      2. A variance-reduction term that rewards evaluations that significantly
         shrink the global posterior uncertainty
      3. Cost normalization to penalize expensive evaluations
    
    The key structural change: instead of just adding a small exploration bonus,
    we compute how much each candidate's evaluation would reduce the total
    posterior variance across the entire candidate set, and use this as a
    principled exploration term.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    t0 = time.time()
    deadline = 300.0  # safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Variance-reduction term: approximate how much evaluating candidate i
    # would reduce the total posterior variance across all candidates.
    # 
    # In a GP, evaluating a point reduces uncertainty in nearby points.
    # We approximate the variance reduction by considering the correlation
    # structure implied by the sigma values. A simple proxy: candidates with
    # high sigma that are "dissimilar" from already well-explored regions
    # (low sigma) provide more information.
    #
    # We use a rank-based approach: the variance-reduction value of candidate i
    # is proportional to how much its evaluation would reduce the sum of
    # variances of all candidates. As a tractable approximation, we use the
    # idea that evaluating a high-uncertainty point in a low-uncertainty
    # neighborhood provides more information than evaluating a high-uncertainty
    # point in a high-uncertainty neighborhood.
    #
    # Proxy: variance_reduction_i = sigma_i * (1 - mean_sigma_nearby)
    # where mean_sigma_nearby is approximated by the local density of low-sigma
    # points. We use a simple kernel-based approximation.
    
    # For tractability, we use a global approximation:
    # The expected variance reduction from evaluating point i is proportional
    # to sigma_i^2 * (1 - rho_i) where rho_i measures how much uncertainty
    # is already reduced in the neighborhood of i.
    
    # Simple proxy: use the ratio of local to global sigma
    # Candidates in regions where sigma is already low (well-explored) get
    # less credit for variance reduction.
    
    # Compute a "local exploration value" using a simple rank-based method
    # Sort candidates by sigma, and assign higher variance-reduction credit
    # to candidates that are outliers in the sigma distribution relative to
    # their neighbors in the mu-sigma space.
    
    # Tractable approximation: variance_reduction_i = sigma_i * (sigma_i / mean_sigma)
    # This rewards candidates with sigma much higher than average.
    
    mean_sigma = np.mean(sigma_safe)
    if mean_sigma < 1e-12:
        mean_sigma = 1e-12
    
    # Variance reduction proxy: higher sigma relative to mean = more information
    # We use sigma_i^2 / mean_sigma as the variance reduction value
    variance_reduction = (sigma_safe ** 2) / mean_sigma
    
    # Normalize variance_reduction to be comparable in scale to EI
    # Scale factor: we want the exploration term to be comparable to EI
    # Use a dynamic scaling based on the current f_best and sigma magnitudes
    scale = max(np.mean(ei), 1e-6)
    variance_reduction_scaled = variance_reduction / (scale + 1e-9)
    
    # Combine EI and variance reduction with a balanced weight
    # The weight controls exploration-exploitation tradeoff
    # Use a moderate weight to ensure both terms contribute
    w_explore = 0.3
    acq = ei + w_explore * variance_reduction_scaled
    
    # Ensure non-negative
    acq = np.maximum(acq, 0.0)
    
    # Cost-aware normalization
    cost_safe = np.maximum(cost, 1e-9)
    scores = acq / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we're close to the limit, return immediately
    if time.time() - t0 > deadline - 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
