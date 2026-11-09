import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (N,) with posterior means.
        sigma: array of shape (N,) with posterior standard deviations.
        f_best: float, the best value observed so far.
        
    Returns:
        array of shape (N,) with scores. Higher scores indicate points 
        that are more likely to be the next best evaluation.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling:
    # We sample a value y_i ~ N(mu_i, sigma_i) for each candidate i.
    # The acquisition score is the probability that this sampled value y_i 
    # is better (lower) than the current best f_best.
    # P(y_i < f_best) = Phi((f_best - mu_i) / sigma_i)
    
    # This is equivalent to using the CDF of the normal distribution.
    # z = (f_best - mu) / sigma
    # score = Phi(z)
    
    # Using log-space for numerical stability if needed, but direct CDF is usually fine
    # for standard double precision unless sigma is extremely small and mu far from f_best.
    # However, if sigma is tiny and mu < f_best, z is large positive, Phi(z) ~ 1.
    # If sigma is tiny and mu > f_best, z is large negative, Phi(z) ~ 0.
    
    z = (f_best - mu) / sigma
    
    # Calculate CDF
    scores = norm.cdf(z)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a small amount of noise to break ties if scores are identical?
    # The evaluator draws 2000 candidates and picks the max score.
    # If multiple points have score 1.0 (very likely if GP is certain they are better),
    # we might want to prefer the one with lower mu (most confident improvement) 
    # or just pick arbitrarily. 
    # Standard Thompson Sampling just picks the max score. 
    # To avoid always picking the same point if there are ties at 1.0, 
    # we can add a tiny perturbation based on mu (lower mu is better).
    # score = Phi(z) + epsilon * (-mu)
    # But the contract says "Return 2000 finite scores, higher meaning evaluate next".
    # The evaluator picks the candidate with the highest score.
    # If scores are tied, behavior might be non-deterministic or index-based.
    # Let's stick to pure Thompson Sampling score.
    
    return scores
