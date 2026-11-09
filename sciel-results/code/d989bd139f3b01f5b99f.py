"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of using a deterministic score like UCB (mu + beta*sigma) or EI,
    we sample a single realization of the function value at each candidate point
    from the Gaussian posterior N(mu, sigma^2).
    
    We then select the candidate with the lowest sampled value (for minimization).
    Since the interface requires returning scores where higher is better,
    we return the negative of the sampled value.
    
    This approach naturally adapts the exploration/exploitation trade-off:
    - Where sigma is high, the sample can be very low (exploration).
    - Where mu is low, the sample is likely low (exploitation).
    """
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from the posterior distribution for each candidate
    # sample_i ~ N(mu_i, sigma_i^2)
    # We use a fixed random state per call? No, global randomness is fine.
    # However, to ensure reproducibility if the evaluator requires it, 
    # standard practice in these loops is often just to use the global RNG.
    # The task says "fresh uniform candidates" are drawn by the evaluator, 
    # implying the evaluator manages its own state, but we manage the acquisition.
    
    samples = mu + sigma * np.random.randn(len(mu))
    
    # We want to minimize the objective, so the "best" candidate has the lowest sample.
    # The acquisition function should return higher scores for better candidates.
    # Therefore, we return -samples.
    score = -samples
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
