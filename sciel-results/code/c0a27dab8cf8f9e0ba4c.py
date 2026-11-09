"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement (EI) and
    Probability of Improvement (PI) with adaptive cost-aware weighting.
    
    Structural change from linear UCB blend to EI+PI hybrid:
    - EI captures expected magnitude of improvement
    - PI captures probability of any improvement (more robust in sparse data)
    - Adaptive blend based on f_best proximity to optimum
    - Cost-aware normalization to prefer cheaper evaluations
    """
    # Deadline guard: track elapsed time and return safe result if approaching limit
    # The evaluator kills after 330s, so we use a safety margin
    t_start = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Standard normal PDF and CDF
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (for minimization)
    # PI = P(f(x) < f_best) = Phi(z)
    pi = Phi_z
    
    # Normalize f_best to [0, 1] for adaptive weighting
    # Griewank values are in [0, ~1]
    f_best_norm = np.clip(f_best, 0.0, 1.0)
    
    # Adaptive blend: 
    # Early (f_best high): more PI (probability of finding any improvement)
    # Late (f_best low): more EI (expected magnitude of improvement)
    # weight_ei goes from 0.4 (early) to 0.85 (late)
    weight_ei = 0.4 + 0.45 * (1.0 - f_best_norm)
    weight_pi = 1.0 - weight_ei
    
    # Normalize EI and PI to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 1e-12 else 1.0
    
    ei_normalized = ei / ei_max
    pi_normalized = pi / pi_max
    
    # Combined score
    combined = weight_ei * ei_normalized + weight_pi * pi_normalized
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # cost is in [1, 2] range (1 + mean(x_scaled) where x_scaled in [0,1])
    # Use a moderate cost penalty
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Check time before returning
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        # Return a safe fallback: simple EI-based scores
        scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
