"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Uses a hybrid of:
    - Expected Improvement (EI) for exploitation
    - Upper Confidence Bound (UCB) for exploration
    - Thompson Sampling-inspired stochastic exploration
    - Cost-awareness via sqrt(cost) normalization
    - Adaptive exploration parameter based on uncertainty and f_best
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB-style exploration term
    # Adaptive beta: higher when uncertainty is high or f_best is far from optimum
    mean_sigma = np.mean(sigma)
    base_beta = 1.8
    if f_best > 50.0:
        base_beta = 2.5
    elif f_best > 10.0:
        base_beta = 2.0
    elif f_best > 1.0:
        base_beta = 1.5
    else:
        base_beta = 1.0
    
    # Increase beta slightly when overall uncertainty is high
    beta = base_beta + 0.5 * np.exp(-mean_sigma * 3.0)
    ucb = -mu + beta * sigma  # Lower mu is better, so we want high -mu + beta*sigma
    
    # Thompson Sampling-inspired: add a stochastic component to encourage exploration
    # Use a fixed seed based on f_best to make this deterministic per call
    # The stochastic component helps escape local minima in multimodal landscapes
    ts_scale = 0.3 * np.exp(-f_best * 0.05)  # Less stochastic as we get closer to optimum
    if ts_scale > 0.01:
        # Create a deterministic pseudo-random perturbation based on mu and sigma
        # This simulates Thompson Sampling without actual randomness
        ts_component = ts_scale * sigma * np.sin(1000.0 * np.tanh(mu / (1.0 + np.abs(mu))))
    else:
        ts_component = np.zeros_like(mu)
    
    # Cost-awareness: use sqrt(cost) for softer normalization
    # This preserves more differentiation between low and high cost points
    cost_norm = np.sqrt(cost)
    
    # Combine EI and UCB with adaptive weighting
    # When f_best is close to 0 (minimum of rastrigin), reduce exploration
    # When f_best is large, increase exploration
    if f_best < 0.5:
        ei_weight = 0.85
        ucb_weight = 0.15
    elif f_best < 2.0:
        ei_weight = 0.75
        ucb_weight = 0.25
    elif f_best < 10.0:
        ei_weight = 0.6
        ucb_weight = 0.4
    elif f_best < 50.0:
        ei_weight = 0.5
        ucb_weight = 0.5
    else:
        ei_weight = 0.4
        ucb_weight = 0.6
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_normalized = (ucb - ucb_min) / ucb_range
    else:
        ucb_normalized = np.zeros_like(ucb)
    ei_normalized = ei / (ei_max + 1e-10)
    
    # Combine
    score = ei_weight * ei_normalized + ucb_weight * ucb_normalized
    
    # Add Thompson Sampling component
    score = score + ts_component / cost_norm
    
    # Apply cost-awareness
    score = score / cost_norm
    
    # Check time budget - return early if needed
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Return current score if we're running out of time
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
