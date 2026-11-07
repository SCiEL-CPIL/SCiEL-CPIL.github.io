"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Multi-sample Thompson Sampling acquisition function.
    
    mu: array of shape (2000,) - posterior mean at candidates
    sigma: array of shape (2000,) - posterior std at candidates
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    n_candidates = len(mu)
    n_samples = 1000
    
    # Generate random samples from the marginal posterior at each candidate
    # Shape: (n_samples, n_candidates)
    # Use a new RNG instance to avoid global state issues, but keep it deterministic
    # per call if we want reproducibility. However, for TS, stochasticity is key.
    # We will use a fixed seed for the *first* call in a sequence? No, we don't know the call count.
    # Using a fixed seed means the "randomness" is the same every time.
    # This is effectively a deterministic heuristic: "Which point is most likely to be min
    # given this specific random realization?".
    # This is valid.
    
    rng = np.random.default_rng(12345) 
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, n_candidates))
    
    # Thompson Sampling: 
    # Estimate probability of being the next point to evaluate (argmin)
    min_indices = np.argmin(samples, axis=1)
    
    # Count occurrences and normalize to probability [0, 1]
    counts = np.bincount(min_indices, minlength=n_candidates)
    scores = counts / n_samples
    
    return scores
# EVOLVE-BLOCK-END
