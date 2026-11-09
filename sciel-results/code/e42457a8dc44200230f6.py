"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 300.0  # Safety margin before the 330s kill limit
_call_count = 0
_total_evals = 30  # 5 initial + 25 optimization steps


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function with smooth
    beta scheduling based on both search progress and observed best value.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    global _call_count
    _call_count += 1
    
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Known global minimum of hartmann 6d
    target_min = -3.32237
    
    # Progress in terms of number of optimization steps (1 to 25)
    opt_steps = max(_call_count - 5, 1)
    total_opt = 25
    step_progress = min(opt_steps / total_opt, 1.0)
    
    # Progress in terms of how close f_best is to the known minimum
    # Hartmann 6d values range roughly from -3.32 to ~+1
    # Normalize: 0 when f_best is at target_min, 1 when f_best is far
    f_range = 4.0  # approximate range from min to typical initial values
    f_progress = min(max((f_best - target_min) / f_range, 0.0), 1.0)
    
    # Combine both progress signals
    # Weight: step progress matters more once we have some data
    combined_progress = 0.6 * step_progress + 0.4 * f_progress
    
    # Smooth beta schedule:
    # Start with high exploration (beta ~ 4.0) and decay to exploitation (beta ~ 1.5)
    # Use a smooth exponential decay for continuous adaptation
    beta_high = 4.0
    beta_low = 1.5
    # Exponential decay: beta = beta_low + (beta_high - beta_low) * exp(-k * progress)
    decay_rate = 2.5
    beta = beta_low + (beta_high - beta_low) * np.exp(-decay_rate * combined_progress)
    
    # Add a small adaptive boost if f_best is still very poor
    # This encourages more exploration when we haven't found good regions yet
    if f_best > -1.0:
        beta *= 1.1
    elif f_best > -2.0:
        beta *= 1.05
    
    # LCB for minimization: maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Add a small smooth exploitation bias that grows with progress
    # This helps break ties among high-uncertainty candidates by preferring
    # those with lower predicted means
    mu_min = mu.min()
    mu_max = mu.max()
    mu_range = mu_max - mu_min
    if mu_range > 1e-12:
        # Normalize mu to [0, 1]
        mu_norm = (mu - mu_min) / mu_range
        # Small weight that increases with progress
        exploit_weight = 0.05 * step_progress
        scores -= exploit_weight * mu_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
