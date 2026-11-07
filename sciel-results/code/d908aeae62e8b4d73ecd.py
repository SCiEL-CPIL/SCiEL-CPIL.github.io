"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Thompson Sampling with multi-sample optimistic LCB hybrid.
    # This structural change moves from deterministic acquisition (EI + exploration terms)
    # to a stochastic sampling approach that better balances exploration/exploitation.
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw multiple samples from the posterior for Thompson Sampling
    # Using 5 samples to reduce variance while maintaining computational efficiency
    n_samples = 5
    rng = np.random.default_rng()
    
    # Generate samples: mu + sigma * standard_normal
    # Shape: (n_samples, 2000)
    samples = mu[None, :] + sigma_safe[None, :] * rng.standard_normal((n_samples, len(mu)))
    
    # For each sample, compute the minimum value (optimistic LCB-like)
    # We want to find the candidate that gives the lowest predicted value
    min_samples = np.min(samples, axis=0)  # Shape: (2000,)
    
    # Also compute the mean of the samples as a stability term
    mean_samples = np.mean(samples, axis=0)
    
    # Blend: primarily use the optimistic (minimum) sample, with some mean contribution
    # The minimum sample provides exploration (pessimistic about what's actually there),
    # while the mean provides exploitation signal
    scores = 0.7 * min_samples + 0.3 * mean_samples
    
    # Invert because we want higher scores for lower predicted values
    scores = -scores
    
    # Add a small exploration bonus based on uncertainty to prevent premature convergence
    # This ensures that regions with high uncertainty still get some attention
    exploration_bonus = 0.1 * sigma_safe
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
