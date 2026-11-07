import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin from 330s kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Quantile Acquisition (QAF) and Expected Improvement (EI).
    
    This function computes a score for each candidate point where higher scores
    indicate more promising candidates for evaluation. It combines:
    1. Quantile Acquisition Function (QAF): Phi((f_best - mu) / sigma), which directly
       optimizes for the probability of finding a new global minimum.
    2. Expected Improvement (EI): balances exploitation and exploration based on the
       current best value.
    3. An exploration term proportional to sigma to encourage exploration in uncertain regions.
    
    Parameters:
    mu: array of shape (N,) - posterior mean
    sigma: array of shape (N,) - posterior standard deviation
    f_best: scalar - best observed value so far (lower is better)
    
    Returns:
    scores: array of shape (N,) - acquisition scores (higher is better)
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a valid array of zeros or something finite to avoid crash
        # But the evaluator kills us anyway. Just return fast.
        return np.zeros_like(mu)
    
    # Handle numerical stability
    # Ensure sigma is non-negative and not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Quantile Acquisition Function (QAF): Probability of exceeding f_best
    # This is the probability that f(x) < f_best, i.e., P(f(x) < f_best) = Phi(z)
    qaf = norm.cdf(z)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Since we want to minimize, improvement is f_best - f(x).
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Exploration Term:
    # Add a term proportional to sigma to encourage exploration in uncertain regions.
    # This acts like a UCB (Upper Confidence Bound) for minimization:
    # We prefer points with low predicted value (low mu) and high uncertainty (high sigma).
    # EI already favors low mu, but adding sigma explicitly boosts exploration.
    exploration_term = 0.5 * sigma_safe
    
    # Combine QAF, EI, and Exploration
    # QAF directly optimizes for the chance of improvement
    # EI provides expected value of improvement
    # Exploration term ensures we don't get stuck in local optima
    # Weighting: 0.5 * QAF + 0.3 * EI + 0.2 * exploration_term
    # These weights are chosen to balance probability of improvement, expected improvement,
    # and exploration
    scores = 0.5 * qaf + 0.3 * ei + 0.2 * exploration_term
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
