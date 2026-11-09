"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Expected Regret normalized by cost for robust multimodal optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _expected_regret(mu, sigma, f_best):
    """
    Compute Expected Regret (ER) for minimization.
    ER = E[max(f_best - f(x), 0)]
    
    For a Gaussian f(x) ~ N(mu, sigma^2):
    ER = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Standard normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Regret
    er = diff * Phi + sigma_safe * phi
    
    # Ensure non-negative (numerical safety)
    er = np.maximum(er, 0.0)
    
    return er

def _kriging_believer(mu, sigma, f_best, cost, alpha=1.0):
    """
    Kriging Believer acquisition function.
    Maximizes: -mu + alpha * sigma (for minimization)
    Then normalizes by cost to account for evaluation cost.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Kriging Believer: maximize -mu + alpha * sigma
    kb = -mu + alpha * sigma_safe
    
    # Cost-aware normalization
    cost_safe = np.maximum(cost, 1e-9)
    scores = kb / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

def _expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) for minimization.
    EI = E[max(f_best - f(x), 0)]
    
    This is the same as Expected Regret in this context.
    """
    return _expected_regret(mu, sigma, f_best)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Uses Expected Regret normalized by cost, with adaptive blending.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        # Return a simple, fast score to avoid timeout
        cost_safe = np.maximum(cost, 1e-9)
        return (-mu + sigma) / cost_safe
    
    # Compute Expected Regret
    er = _expected_regret(mu, sigma, f_best)
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    er_scores = er / cost_safe
    
    # Adaptive alpha for Kriging Believer:
    # - When f_best is high (early), use larger alpha for more exploration
    # - When f_best is low (late), use smaller alpha for more exploitation
    if f_best > 10.0:
        alpha = 2.0
    elif f_best > 5.0:
        alpha = 1.5
    elif f_best > 2.0:
        alpha = 1.0
    elif f_best > 1.0:
        alpha = 0.7
    else:
        alpha = 0.5
    
    kb_scores = _kriging_believer(mu, sigma, f_best, cost, alpha=alpha)
    
    # Blend ER and KB scores:
    # - Early on (high f_best), blend more towards KB for exploration
    # - Later (low f_best), blend more towards ER for exploitation
    if f_best > 10.0:
        blend = 0.4  # 40% ER, 60% KB
    elif f_best > 5.0:
        blend = 0.5
    elif f_best > 2.0:
        blend = 0.6
    elif f_best > 1.0:
        blend = 0.7
    else:
        blend = 0.8  # 80% ER, 20% KB
    
    # Normalize both score sets to [0, 1] range for fair blending
    er_max = np.max(er_scores)
    kb_max = np.max(kb_scores)
    
    if er_max > 1e-12:
        er_normalized = er_scores / er_max
    else:
        er_normalized = np.zeros_like(er_scores)
    
    if kb_max > 1e-12:
        kb_normalized = kb_scores / kb_max
    else:
        kb_normalized = np.zeros_like(kb_scores)
    
    # Blend the normalized scores
    scores = blend * er_normalized + (1.0 - blend) * kb_normalized
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
