"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Monte Carlo approximation of Expected Regret relative to the known
    global minimum (3.0) of the Goldstein-Price function.
    
    mu: (2000,) array of GP posterior means
    sigma: (2000,) array of GP posterior standard deviations
    f_best: lowest observed value so far
    
    Returns: (2000,) array of scores, higher = more promising to evaluate next.
    """
    # Known global minimum of Goldstein-Price 2D on [-2,2]^2 is 3.0
    f_true_min = 3.0
    
    n = len(mu)
    # Number of Monte Carlo samples for estimating the expected regret
    # 100 samples provide a good trade-off between accuracy and speed
    K = 100
    
    # Generate K standard normal samples for each of the n candidates
    # Shape: (K, n)
    Z = np.random.randn(K, n)
    
    # Compute K samples of the function value for each candidate
    # f(x) ~ N(mu(x), sigma(x)^2)
    # Broadcasting: mu[None, :] is (1, n), sigma[None, :] is (1, n)
    # Resulting shape: (K, n)
    samples = mu[None, :] + sigma[None, :] * Z
    
    # Calculate the regret for each sample: max(0, f_true_min - f_sample)
    regrets = np.maximum(0.0, f_true_min - samples)
    
    # Average the regret over the Monte Carlo samples to get the expected regret
    scores = np.mean(regrets, axis=0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
