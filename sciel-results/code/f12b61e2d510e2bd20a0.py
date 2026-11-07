import numpy as np
import time
from scipy.stats import norm

# Deadline guard: ensure we finish well before 330s wall-clock
_DEADLINE = time.time() + 290.0  # safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling with a small deterministic
    exploitation bias.
    
    Strategy:
      - Sample y ~ N(mu, sigma) for each candidate (Thompson Sampling).
      - Score = (f_best - y) / cost  (expected regret reduction per unit cost).
      - Add a small deterministic EI/cost term to stabilize selection and
        reduce variance of the stochastic score.
    
    This stochastically balances exploration and exploitation without requiring
    complex normalization or heuristic beta parameters.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # --- Thompson Sampling score ---
    # Sample one function value per candidate
    y_sample = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # Regret reduction per unit cost (higher is better)
    ts_score = (f_best - y_sample) / cost_safe
    
    # --- Deterministic EI/cost component for stability ---
    # z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_score = ei / cost_safe
    
    # Combine: primarily Thompson Sampling, with a small EI bias
    # The EI term helps when the stochastic sample is uninformative
    # (e.g., very early on when sigma is large and samples are noisy)
    w_ts = 0.85
    w_ei = 0.15
    
    score = w_ts * ts_score + w_ei * ei_score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
