import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.5  # 1.5 second safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in the tails
    z = np.clip(z, -50, 50)
    
    # Expected Improvement for minimization
    # E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Structural enhancement: Monte Carlo estimated probability of being the global minimum
    # This helps avoid local optima by boosting candidates that are likely to be the best
    # in the entire search space, not just better than the current best.
    
    # Estimate the probability that each candidate is the global minimum
    # using a Monte Carlo approach with a small number of samples
    
    n_candidates = len(mu)
    
    # Use a quick Monte Carlo estimate of the probability that each point
    # has the lowest value among all candidates
    # We sample a small number of times to keep it fast
    
    n_mc_samples = 50  # Keep small for speed
    
    # Generate random samples from the GP posterior
    # For each sample, we get a set of function values
    # We count how often each candidate has the minimum value
    
    # To keep this fast, we'll use a simplified approach:
    # Estimate P(f(x_i) = min(f(x_j))) using the GP posterior
    
    # A more efficient approach: use the fact that for a GP,
    # we can estimate the probability of being the minimum using
    # the mean and covariance structure
    
    # For simplicity and speed, we'll use a Monte Carlo approach with
    # a small number of samples
    
    # Generate MC samples from the GP posterior
    # x ~ N(mu, sigma^2) for each candidate independently
    # (Note: this ignores correlations, but it's a reasonable approximation
    #  and much faster than full GP sampling)
    
    # Actually, let's use a simpler but effective approach:
    # Estimate the probability that each candidate is better than all others
    # using pairwise comparisons or a quick MC estimate
    
    # Quick MC estimate: sample a few times and count minima
    mc_counts = np.zeros(n_candidates)
    
    # Use a fixed random seed for reproducibility within the call
    rng = np.random.default_rng(42)
    
    for _ in range(n_mc_samples):
        # Sample function values from the GP posterior
        f_samples = mu + sigma_safe * rng.standard_normal(n_candidates)
        
        # Find the index of the minimum
        min_idx = np.argmin(f_samples)
        mc_counts[min_idx] += 1
    
    # Probability of being the minimum
    p_min = mc_counts / n_mc_samples
    
    # Boost the EI score by the probability of being the global minimum
    # This encourages exploration of regions that are likely to contain
    # the global minimum, not just points that might improve over f_best
    
    # The boost should be scaled appropriately
    # A small boost to avoid overwhelming the EI signal
    boost = 0.1 * p_min
    
    # Combine EI with the boost
    scores = ei + boost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
