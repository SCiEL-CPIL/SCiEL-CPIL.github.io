"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    if _check_deadline():
        # Fallback: return deterministic EI if we're running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Thompson Sampling: sample from posterior, weighted by cost
    # For minimization, we want low values, so we sample y ~ N(mu, sigma^2)
    # and prefer candidates with low y and low cost.
    # Acquisition score = -y_sample / cost (higher is better for selection)
    
    # Add small epsilon to avoid division by zero
    cost_safe = cost + 1e-9
    sigma_safe = sigma + 1e-9
    
    # Draw a sample from the posterior
    # Use a fixed random seed per call? No, we want stochasticity.
    # But we need reproducibility within a run? The task doesn't require it.
    # However, for stability, let's use a seeded RNG based on call count or just random.
    # Using np.random directly.
    sample = mu - sigma_safe * np.random.randn(len(mu))
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    # We want to maximize this score (higher = evaluate next)
    # Since we're minimizing f, we want low sample values.
    # So score = -sample / cost
    scores = -sample / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
