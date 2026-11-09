import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Probabilistic Lower Bound (PLB) for minimization:
    score = -mu + k * sigma_eff, where k adapts to the current gap from optimum.
    """
    start_time = time.time()
    timeout = 320.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive for computations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Gap from optimum determines exploration-exploitation tradeoff
    gap = max(f_best - 3.0, 0.0)
    
    # Dynamic k: when far from optimum, explore more (larger k);
    # when close, exploit more (smaller k).
    # Use a smoother decay: k = k_min + (k_max - k_min) * (1 + gap/k_scale)^(-alpha)
    # This gives a more gradual transition than pure exponential.
    k_min = 0.3
    k_max = 3.5
    k_scale = 0.8
    alpha = 0.7
    k = k_min + (k_max - k_min) * (1.0 + gap / k_scale) ** (-alpha)
    
    # Smooth sigma damping using a log-sum-exp inspired soft cap:
    # sigma_eff = sigma_scale * log1p(sigma_safe / sigma_scale)
    # This grows logarithmically for large sigma, preventing extreme exploration
    # while still allowing meaningful differentiation for small sigma.
    sigma_scale = 1.5
    sigma_eff = sigma_scale * np.log1p(sigma_safe / sigma_scale)
    
    # PLB score for minimization: -mu + k * sigma_eff
    scores = -mu + k * sigma_eff
    
    # Exploitation bonus: when close to optimum, reward points with low mu
    # Use a smooth sigmoid-like transition rather than a hard threshold
    # bonus_weight ramps up as gap decreases
    if gap < 2.0:
        # Smooth weight: 0 at gap=2, approaches 1 as gap->0
        bonus_weight = 0.25 * (1.0 - gap / 2.0) ** 2
        # Normalize improvement relative to gap
        improvement = np.maximum(0.0, f_best - mu)
        mu_bonus = improvement / max(gap, 0.1)
        # Soft-cap the bonus to prevent dominance
        mu_bonus = np.tanh(mu_bonus) * bonus_weight
        scores += mu_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - start_time > timeout:
        return scores
    
    return scores
