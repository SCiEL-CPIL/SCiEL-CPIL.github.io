"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-12

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Strategy:
    Instead of computing a deterministic score (like EI or UCB) and picking the max,
    we sample a function from the GP posterior at the candidate points.
    For each candidate x_i, we sample y_i ~ N(mu_i, sigma_i^2).
    The "acquisition score" is the negative of this sampled value (-y_i),
    because we want to minimize the objective. By taking the max of these scores,
    the loop effectively selects the candidate that has the highest probability
    of being the minimum, weighted by uncertainty.
    
    This is a structural change from deterministic acquisition functions to a
    stochastic search method. It is known to be more robust in multimodal
    landscapes with limited budgets because it naturally explores high-uncertainty
    regions without requiring manual tuning of exploration constants (kappa).
    """
    start_time = time.time()
    
    n = len(mu)
    
    # Ensure inputs are valid
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        # Fallback to a simple UCB if inputs are corrupted
        return -mu + 2.0 * sigma
    
    # Thompson Sampling: Sample from the posterior N(mu, sigma^2)
    # We use a fixed random state for reproducibility within the call if needed,
    # but since the evaluator runs multiple steps, we just draw fresh samples.
    # Note: The problem statement says "draws 2000 fresh uniform candidates", 
    # implying the caller generates new x's each step. We just need to return scores.
    
    # Draw samples
    # sigma is standard deviation, so we multiply by standard normal randoms
    samples = mu + sigma * np.random.standard_normal(n)
    
    # For minimization, we want to find the minimum.
    # Thompson Sampling selects the argmin of the samples.
    # The evaluator picks the candidate with the "highest score".
    # So we return -samples to make the minimum the maximum score.
    scores = -samples
    
    # Safety: ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Optional: Add a small deterministic bias towards exploitation if the posterior
    # is very confident? No, pure TS is the structural change.
    
    return scores

# EVOLVE-BLOCK-END
