"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Independent Thompson Sampling.
    
    This method samples from the GP posterior at each candidate point independently.
    By sampling $f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$ and selecting the point
    with the lowest sampled value, we naturally balance exploration (high $\sigma$ 
    allows for potentially very low samples) and exploitation (low $\mu$ leads to 
    low samples). This is structurally different from deterministic methods like 
    EI or UCB and often performs well in small-budget settings by avoiding 
    premature convergence to local minima.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far. (Not directly used in TS, 
        as the posterior already reflects the data).
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
        Here, score = -sampled_value.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 290.0  # 10s safety margin before 300s limit
    
    # Ensure sigma is positive to avoid issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw random samples from the standard normal distribution
    # Using np.random.normal for vectorized operation
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior distribution: f ~ N(mu, sigma^2)
    # f_sampled = mu + sigma * z
    f_sampled = mu + sigma_safe * z
    
    # We are minimizing the function. 
    # The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    scores = -f_sampled
    
    # Check deadline (TS is O(N) and very fast, so this is unlikely to trigger)
    if time.time() > deadline:
        # Fallback: return a deterministic score based on mean if time is critically low
        # Though for 2000 points, the above should take < 1ms
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
