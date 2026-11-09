"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Thompson Sampling.
    
    Instead of computing a deterministic improvement metric like EI or PoI,
    we sample from the posterior distribution of the objective function at each
    candidate point. The point with the lowest sampled value (adjusted for cost)
    is selected. This naturally balances exploration (high sigma -> wide sample
    range) and exploitation (low sigma -> sample near mu).
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Check deadline early - if close to timeout, fall back to a simple heuristic
    if _check_deadline():
        # Simple fallback: minimize mu / cost
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Thompson Sampling: sample from N(mu, sigma^2) for each candidate
    # We use a fixed seed per call to ensure reproducibility within a run,
    # but different calls will have different samples due to the random state
    # Actually, for Thompson sampling in a loop, we want fresh randomness each time.
    # NumPy's global random state is fine here.
    
    # Generate random samples from the posterior
    # sample = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.randn(mu.shape[0])
    samples = mu + sigma_safe * z
    
    # For minimization, we want to pick points with low sampled values.
    # The acquisition score should be higher for points we want to evaluate.
    # So we negate the sample: higher score = lower sampled value.
    
    # Cost-aware adjustment: divide by cost to account for evaluation expense.
    # A point with a very low sample but high cost might be less attractive
    # than a point with a slightly higher sample but much lower cost.
    
    # The score is -sample / cost
    # This encourages points that are likely to be good (low sample) and cheap.
    
    scores = -samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
