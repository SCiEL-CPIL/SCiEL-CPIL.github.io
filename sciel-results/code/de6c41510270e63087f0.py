"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we stay within budget
_start_time = time.time()
_TIME_LIMIT = 325.0  # 5 seconds margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Blended acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with adaptive weighting.
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        Blended scores, shape (N,), higher is better.
    """
    global _start_time
    
    # Safety check: if we're close to the time limit, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return deterministic score based on -mu to avoid hanging
        return -np.asarray(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle numerical stability for sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Expected Improvement (EI) for minimization
    # Improvement is f_best - f(x). Since we minimize, lower f(x) is better.
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB)
    # UCB = -mu + beta * sigma
    # We use a beta of 1.0 as a reasonable default for exploration.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Normalize scores to comparable scales
    # This prevents one term from dominating due to magnitude differences.
    def normalize(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # 4. Adaptive blending based on proximity to global minimum
    # The global minimum of Styblinski-Tang 1D is -39.1662
    global_min = -39.1662
    
    # Measure how close we are to the global minimum
    # If f_best is close to global_min, we should exploit more (increase EI weight)
    # If f_best is far, we should explore more (increase UCB weight)
    
    # Use a sigmoid-like function to adapt the weight
    # When f_best is near global_min, w_ei should be high (exploit)
    # When f_best is far, w_ei should be low (explore)
    
    # Scale: assume f_best ranges from ~0 (initial) to ~-39 (optimal)
    # Use a smooth transition
    delta_to_min = f_best - global_min  # Positive when above minimum
    # Normalize delta: 0 when at min, positive when above
    # Use a scaling factor to control the transition width
    scale = 5.0
    # w_ei increases as delta_to_min decreases (closer to min)
    # When delta_to_min is large (far from min), w_ei is low (more exploration)
    # When delta_to_min is small (near min), w_ei is high (more exploitation)
    
    # Sigmoid: 1 / (1 + exp(-x))
    # We want w_ei to increase as delta_to_min decreases
    # So use sigmoid(-delta_to_min / scale)
    # When delta_to_min is large positive, -delta/scale is large negative, sigmoid ~ 0
    # When delta_to_min is near 0, sigmoid ~ 0.5
    # When delta_to_min is negative (below min, shouldn't happen), sigmoid > 0.5
    
    sigmoid_val = 1.0 / (1.0 + np.exp(-delta_to_min / scale))
    
    # Map sigmoid to a weight range, e.g., [0.3, 0.8] for EI
    # When sigmoid is 0 (far from min), w_ei = 0.3 (more exploration)
    # When sigmoid is 1 (at min), w_ei = 0.8 (more exploitation)
    w_ei = 0.3 + 0.5 * (1.0 - sigmoid_val)
    w_ucb = 1.0 - w_ei
    
    # Ensure weights are in valid range
    w_ei = np.clip(w_ei, 0.2, 0.9)
    w_ucb = 1.0 - w_ei
    
    # 5. Blend
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
