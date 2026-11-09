# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    deadline = t_start + 295.0  # Safety margin for 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is safe
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Thompson Sampling:
    # Draw a sample from the posterior predictive distribution at each candidate.
    # The posterior predictive is Gaussian: N(mu, sigma^2).
    # We draw one sample: x_i ~ N(mu_i, sigma_i^2)
    # Since we want to MINIMIZE the objective, a low sample value indicates a promising region.
    # The task requires returning scores where HIGHER means evaluate next.
    # Therefore, we return the negative of the sampled values.
    
    # To ensure the random draw is reproducible within a single run but different across runs/seeds,
    # we rely on the global numpy random state which is typically seeded by the evaluator framework
    # or left to default. If the evaluator seeds np.random, this will be consistent.
    # If not, it's still a valid random strategy.
    
    # Draw samples
    samples = mu - sigma_safe * np.random.standard_normal(n)
    
    # Convert minimization problem to maximization for scoring
    scores = -samples
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Time check (though this is very fast)
    if time.time() > deadline:
        pass
        
    return scores

# EVOLVE-BLOCK-END
