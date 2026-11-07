"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    KG estimates the expected reduction in the global minimum value.
    For each candidate i, KG_i = E[max(0, f_best - min_j Y_j)]
    where Y_j are the predicted values at all candidates.
    
    We use a fast approximation:
    1. Compute EI for each point
    2. Estimate the probability that point i is the minimum among all candidates
    3. KG_i = EI_i * P(i is min)
    
    P(i is min) is approximated using a softmax over the negative standardized values.
    """
    t0 = time.time()
    deadline = 320.0  # Leave 10s margin before 330s kill
    
    n = len(mu)
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardize
    d = (f_best - mu) / sigma_safe
    
    # Compute EI for minimization
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    EI = (f_best - mu) * Phi_d + sigma_safe * phi_d
    EI = np.maximum(EI, 0.0)
    
    # Check time
    if time.time() - t0 > deadline:
        return np.nan_to_num(EI, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Estimate the probability that point i is the global minimum
    # Use a softmax-like weighting based on how "good" each point is
    # Points with low mu and moderate sigma are more likely to be the minimum
    
    # Approximation: P(i is min) ∝ exp(-mu_i / (sigma_i + eps))
    # This gives higher probability to points with low mu
    
    # For numerical stability, subtract the maximum
    log_weights = -mu / (sigma_safe + 1e-10)
    log_weights_max = np.max(log_weights)
    log_weights = log_weights - log_weights_max
    
    weights = np.exp(log_weights)
    sum_weights = np.sum(weights)
    if sum_weights < 1e-15:
        sum_weights = 1e-15
    weights = weights / sum_weights
    
    # KG score: EI weighted by probability of being the minimum
    KG = EI * weights
    
    # Add a small exploration bonus to encourage exploring uncertain regions
    # This helps in the early stages when f_best is still high
    exploration_bonus = 0.05 * sigma_safe / (abs(f_best) + 1e-10)
    KG = KG + exploration_bonus
    
    # Check time
    if time.time() - t0 > deadline:
        KG = np.nan_to_num(KG, nan=0.0, posinf=0.0, neginf=0.0)
        return KG
    
    # Ensure finite scores
    KG = np.nan_to_num(KG, nan=0.0, posinf=0.0, neginf=0.0)
    
    return KG
# EVOLVE-BLOCK-END
