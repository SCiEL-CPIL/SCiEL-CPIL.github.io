"""Cost-aware acquisition for rastrigin 2d with QI-based strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile of Improvement (QI) based cost-aware acquisition function.
    
    QI focuses on the conditional expected value given improvement,
    balancing exploration and exploitation in multimodal landscapes.
    We combine QI with a small exploration bonus for robustness.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Z-score: how many std devs below f_best
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    # QI: Conditional expected improvement given that improvement occurs
    # E[f_best - f | f < f_best] * P(f < f_best) / P(f < f_best) 
    # Actually, QI = E[(f_best - f)^+ * 1{f < f_best}] / P(f < f_best)
    # For a normal distribution:
    # P(f < f_best) = Phi(z)
    # E[(f_best - f)^+ * 1{f < f_best}] = sigma * [z * Phi(z) + phi(z)]
    # So QI = sigma * [z * Phi(z) + phi(z)] / Phi(z)
    #      = sigma * [z + phi(z)/Phi(z)]
    
    pi = norm.cdf(z)
    pi_safe = np.maximum(pi, 1e-12)
    
    # phi(z) = pdf
    phi_z = norm.pdf(z)
    
    # QI = sigma * (z + phi(z)/pi)
    qi = sigma_safe * (z + phi_z / pi_safe)
    
    # Clip QI to non-negative (should be, but numerical issues can arise)
    qi = np.maximum(qi, 0.0)
    
    # Add a small exploration bonus: sigma / cost
    # This encourages visiting high-uncertainty regions
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine QI and exploration bonus, normalized by cost
    score = (qi + exploration_bonus) / cost_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
