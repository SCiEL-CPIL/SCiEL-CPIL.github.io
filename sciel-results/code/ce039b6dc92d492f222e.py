"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Structural change: Uses a cost-aware Probability of Improvement (PI) 
    combined with an adaptive exploration term that scales with the 
    information gain (sigma) relative to cost. This properly models the 
    trade-off between the probability of finding a better solution and 
    the evaluation cost.
    
    Key insight: For minimization with cost, we want to maximize the 
    probability of improvement per unit cost. PI = P(f(x) < f_best) = 
    Phi((f_best - mu) / sigma). The cost-adjusted score is PI / cost, 
    which directly optimizes the probability of success per cost unit.
    """
    if not _time_ok():
        # Fallback: return cost-adjusted EI quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    pi_prob = norm.cdf(z)
    pi_prob = np.clip(pi_prob, 1e-15, 1.0 - 1e-15)
    
    # Expected Improvement
    ei = (f_best - mu) * pi_prob + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring:
    # Primary: Probability of Improvement per unit cost
    # This directly optimizes the chance of finding a better point given the cost
    pi_cost = pi_prob / (cost + 1e-9)
    
    # Secondary: Expected Improvement per unit cost (for exploitation)
    ei_cost = ei / (cost + 1e-9)
    
    # Tertiary: Information gain per unit cost (for exploration)
    # Points with high sigma relative to cost provide more information
    info_gain = sigma_safe / (cost + 1e-9)
    
    # Adaptive blending:
    # Early on (high uncertainty), favor information gain
    # Later (low uncertainty), favor probability of improvement
    mean_sigma = np.mean(sigma_safe)
    if mean_sigma > 0.1:
        # High uncertainty: more exploration
        w_pi = 0.3
        w_ei = 0.4
        w_info = 0.3
    elif mean_sigma > 0.01:
        # Medium uncertainty: balanced
        w_pi = 0.4
        w_ei = 0.4
        w_info = 0.2
    else:
        # Low uncertainty: exploitation
        w_pi = 0.5
        w_ei = 0.5
        w_info = 0.0
    
    score = w_pi * pi_cost + w_ei * ei_cost + w_info * info_gain
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
