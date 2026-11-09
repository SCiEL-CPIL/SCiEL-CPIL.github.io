"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    We approximate the Expected Improvement (EI) per unit cost using Monte Carlo
    sampling from the GP posterior.
    
    1. Draw N samples from N(mu, sigma^2) for each candidate.
    2. Compute the improvement over f_best for each sample: max(0, f_best - sample).
    3. Average these improvements to estimate EI.
    4. Divide by the evaluation cost to get the expected improvement per cost.
    
    This balances exploration (high sigma leads to higher potential EI) and
    exploitation (high f_best - mu leads to higher EI) while penalizing expensive
    evaluations.
    """
    n_candidates = len(mu)
    n_samples = 20  # Number of Monte Carlo samples for stable EI estimation
    
    # Reshape for broadcasting: (n_samples, n_candidates)
    mu_b = mu.reshape(1, -1)
    sigma_b = sigma.reshape(1, -1)
    
    # Draw samples: (n_samples, n_candidates)
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu_b, scale=sigma_b, size=(n_samples, n_candidates))
    
    # Calculate improvement over the current best
    # f_best is a scalar (lowest observed value)
    # We want positive values if the sample is better (lower) than f_best
    improvements = f_best - samples  # (n_samples, n_candidates)
    
    # Only count positive improvements
    positive_improvements = np.maximum(0, improvements)
    
    # Estimate Expected Improvement (EI)
    ei = np.mean(positive_improvements, axis=0)
    
    # Normalize by cost
    # cost has shape (n_candidates,)
    scores = ei / cost
    
    # Ensure finite values
    # If cost is 0 (shouldn't happen based on task desc: 1 + ...), handle it
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
