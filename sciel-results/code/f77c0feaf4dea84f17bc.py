"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for Bayesian optimization of Ackley 2D.
    
    Draws a single sample from the posterior predictive distribution
    f(x) ~ N(mu, sigma) for each candidate and uses -sample as the score.
    This naturally balances exploration and exploitation:
    - Points with low mu (good predictions) get high scores
    - Points with high sigma (uncertain) can occasionally get high scores
    - The balance adapts automatically as uncertainty decreases
    """
    # Check deadline
    if _time_check():
        # Return a simple fallback: negative mean (pure exploitation)
        mu_safe = np.asarray(mu, dtype=np.float64)
        mu_safe = np.where(np.isfinite(mu_safe), mu_safe, 0.0)
        return -mu_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw one sample from N(mu, sigma) for each candidate
    # Use a fixed seed based on f_best to make it deterministic per f_best value
    # This ensures reproducibility while still getting stochastic behavior across steps
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 1.0
    
    # Create a deterministic but varying seed based on f_best
    # This gives different samples for different f_best values
    seed = int(abs(f_best * 10000)) % (2**32)
    rng = np.random.default_rng(seed)
    
    # Draw samples from the posterior predictive
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: negative sample (we want to minimize, so higher score = lower predicted value)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
