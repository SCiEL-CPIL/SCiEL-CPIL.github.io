import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    Samples from the GP posterior and returns the negative sample as the score.
    This encourages exploration where uncertainty is high and exploitation
    where the mean is low, without explicit hyperparameters.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0  # 1 second safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Floor sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: Draw one sample from the posterior for each candidate
    # y_i ~ N(mu_i, sigma_i^2)
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(n)
    
    # We are minimizing f. We want to select the candidate with the lowest f.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    score = -samples
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline (should not trigger for 2000 points)
    if time.time() > deadline:
        # Fallback to deterministic EI if it's somehow too slow (unlikely)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        from scipy.special import erfc
        sqrt_2 = np.sqrt(2.0)
        sqrt_2pi = np.sqrt(2.0 * np.pi)
        Phi = 0.5 * erfc(-z_clipped / sqrt_2)
        phi = np.exp(-0.5 * z_clipped**2) / sqrt_2pi
        EI = (f_best - mu) * Phi + sigma_safe * phi
        EI = np.maximum(EI, 0.0)
        score = EI
        
    return score
