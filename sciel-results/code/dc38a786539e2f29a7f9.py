# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for minimization of 1D sphere.
    Samples from the posterior predictive distribution and selects the
    candidate with the lowest sampled value. This naturally balances
    exploration (high sigma leads to large variance in samples) and
    exploitation (low mu leads to low samples).
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Thompson Sampling: sample once from the posterior predictive
    # For minimization, we want to select the point with the lowest
    # sampled value. The sample is mu + sigma * z where z ~ N(0,1).
    
    # Generate a single random standard normal draw
    # Using a fixed seed per call would make it deterministic, but we want
    # stochastic behavior. However, for reproducibility in the evaluator,
    # we just use np.random which is seeded by the environment.
    z = np.random.standard_normal(mu.shape)
    
    # The sampled objective values
    samples = mu + sigma * z
    
    # For minimization, we want to pick the point with the LOWEST sample.
    # So the acquisition score should be HIGH for points with LOW samples.
    # Therefore, score = -samples
    score = -samples

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)

    # Deadline guard: if computation took too long (unlikely for this simple op),
    # return a safe fallback. In practice, this is extremely fast.
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple UCB-like score
        beta = 1.0
        return -mu + beta * sigma

    return score
# EVOLVE-BLOCK-END
