"""Rewritten program with Hybrid Adaptive UCB/EI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid Adaptive UCB/EI acquisition function.
    
    Blends Upper Confidence Bound (UCB) and Expected Improvement (EI)
    based on the current best value. Uses a lighter cost penalty (sqrt)
    to avoid over-penalizing the search region near the optimum.
    """
    t_start = time.time()
    
    # Safety checks
    sigma_safe = np.clip(sigma, 1e-8, None)
    cost_safe = np.clip(cost, 1.0, None)
    
    # 1. Compute EI
    delta = f_best - mu
    z = delta / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    EI = delta * Phi_z + sigma_safe * phi_z
    EI = np.maximum(EI, 0.0)
    
    # 2. Compute Adaptive UCB
    # Beta decays as f_best approaches 0 (the minimum for Rosenbrock)
    # When f_best is large (early), beta is high (explore)
    # When f_best is small (late), beta is low (exploit)
    # Using exponential decay with a floor of 1.0
    beta = 1.0 + 3.0 * np.exp(-0.5 * np.maximum(f_best, 0.0))
    
    # UCB for minimization: -mu + beta * sigma
    # We maximize this score, so we want high beta*sigma and low mu
    UCB = -mu + beta * sigma_safe
    
    # 3. Cost Normalization
    # Use sqrt(cost) which is less aggressive than cost^0.75
    cost_penalty = np.sqrt(cost_safe)
    
    score_ei = EI / cost_penalty
    score_ucb = UCB / cost_penalty
    
    # 4. Adaptive Blending
    # Weight for UCB: higher when f_best is large
    # Sigmoid centered around f_best = 1.0 (typical early value for Rosenbrock shifted)
    # If f_best > 1, we weight UCB more heavily.
    # If f_best < 1, we weight EI more heavily.
    w_ucb = 1.0 / (1.0 + np.exp(-1.5 * (f_best - 1.0)))
    w_ei = 1.0 - w_ucb
    
    score = w_ucb * score_ucb + w_ei * score_ei
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t_start > 325:
        # In this simple case, we just return what we have.
        # Complex operations would go in a loop, but here it's vectorized.
        pass

    return score
# EVOLVE-BLOCK-END
