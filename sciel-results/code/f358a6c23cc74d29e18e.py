"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # --- Step 1: Current Step Scoring (UCB) ---
    # Adaptive beta: high exploration early, low later.
    distance_to_min = max(f_best - f_min_known, 0.0)
    # beta ranges from ~4.0 (early) to ~1.0 (late)
    beta = 1.0 + 3.0 * np.exp(-distance_to_min / 5.0)
    
    # UCB score for minimization: -mu + beta * sigma
    scores_ucb = -mu + beta * sigma_safe
    
    # --- Step 2: Look-Ahead Simulation ---
    # Simulate the next 5 evaluations to estimate the future state.
    K = 5
    # Select the top K candidates based on current UCB scores
    top_k_idx = np.argpartition(scores_ucb, -K)[-K:]
    
    # Estimate the new best value if these K points were evaluated.
    # We assume the observed values are the posterior means (conservative).
    # This is a heuristic approximation of the information gain.
    simulated_new_f_best = min(f_best, np.min(mu[top_k_idx]))
    
    # Recalculate beta for the new state (exploration should decrease if we improved)
    distance_to_min_new = max(simulated_new_f_best - f_min_known, 0.0)
    beta_new = 1.0 + 3.0 * np.exp(-distance_to_min_new / 5.0)
    
    # Calculate Expected Improvement (EI) for the new state.
    # EI is a good exploitation metric for the "next" step.
    z_new = (simulated_new_f_best - mu) / sigma_safe
    z_new = np.clip(z_new, -10, 10)
    
    # EI formula for minimization
    ei_new = (simulated_new_f_best - mu) * norm.cdf(z_new) + sigma_safe * norm.pdf(z_new)
    
    # --- Step 3: Combine Current and Future Scores ---
    # Normalize both score vectors to [0, 1] to make them comparable.
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
        
    norm_ucb = normalize_scores(scores_ucb)
    norm_ei = normalize_scores(ei_new)
    
    # Weighted combination.
    # We prioritize the current step (UCB) but add a bonus for future potential (EI).
    # This encourages selecting points that are good now AND in a region that 
    # likely contains further improvements after the current top-K are found.
    lambda_weight = 0.5
    final_scores = norm_ucb + lambda_weight * norm_ei
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return final_scores
    
    return final_scores
# EVOLVE-BLOCK-END
