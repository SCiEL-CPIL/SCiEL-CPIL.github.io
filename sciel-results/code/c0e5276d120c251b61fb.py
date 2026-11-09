"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
import numpy as np
import time
import math


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses a cost-adjusted Expected Improvement (EI) variant with adaptive
    exploration. The key idea is to balance the expected improvement over
    f_best against the evaluation cost, while maintaining sufficient
    exploration to find the global minimum.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions
    sigma : array_like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed function value
    cost : array_like, shape (2000,)
        Evaluation cost for each candidate (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    t_start = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Convert to arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle numerical issues
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=10.0, neginf=1.0)
    
    # Ensure valid ranges
    sigma = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-9)
    
    n = len(mu)
    
    # --- Adaptive exploration parameter ---
    # Estimate progress: use the distribution of mu relative to f_best
    # If most candidates have mu close to or below f_best, we're in exploitation mode
    # If many have mu >> f_best, we should explore more
    
    mu_below = np.sum(mu < f_best)
    mu_ratio = mu_below / n if n > 0 else 0.5
    
    # Also consider the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    
    # High sigma relative to the gap between f_best and typical mu suggests exploration
    mu_median = np.median(mu)
    gap = f_best - mu_median  # Positive if f_best is below median mu
    
    # Exploration intensity: higher when sigma is large relative to the gap
    if gap > 1e-9:
        exploration_intensity = min(sigma_mean / (gap + 1e-9), 3.0)
    else:
        exploration_intensity = 2.0
    
    # Budget-aware adjustment: if costs are generally high, we've spent more budget
    cost_mean = np.mean(cost_safe)
    # cost is in [1, 2] typically (1 + mean of 4 coords in [0,1])
    # Average cost ~1.5, so if cost_mean is high, we're later in the budget
    budget_pressure = (cost_mean - 1.0) / 1.0  # 0 to 1
    
    # Dynamic beta: start with higher exploration, taper off
    # beta controls the weight of sigma in the acquisition
    base_beta = 1.8 + 1.2 * min(exploration_intensity, 2.5)
    beta = base_beta * (1.0 - 0.4 * budget_pressure)
    beta = np.clip(beta, 0.8, 3.5)
    
    # --- Cost-Adjusted Expected Improvement ---
    # For minimization: improvement = f_best - f(x)
    # E[improvement] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the PDF
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    # Use scipy if available, otherwise approximate
    try:
        from scipy.special import ndtr, ndtrf
        Phi = ndtr(z_clipped)  # CDF
        # PDF: phi(z) = exp(-z^2/2) / sqrt(2*pi)
        phi = np.exp(-0.5 * z_clipped**2) / np.sqrt(2.0 * np.pi)
    except ImportError:
        # Fallback: use math.erf for CDF approximation
        Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z_clipped / np.sqrt(2.0)))
        phi = np.exp(-0.5 * z_clipped**2) / np.sqrt(2.0 * np.pi)
    
    # Expected improvement (for minimization)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # --- Cost-adjusted score ---
    # We want high EI and low cost
    # score = EI / cost^alpha, where alpha controls cost sensitivity
    # alpha=1 is linear cost penalty, alpha=0.5 is softer
    
    # Adaptive alpha: when EI values are spread out, use stronger cost penalty
    ei_range = np.ptp(ei) + 1e-9
    ei_std = np.std(ei)
    
    # If EI is very concentrated, cost matters less; if spread out, cost matters more
    alpha = 0.7 + 0.3 * min(ei_std / (ei_range + 1e-9), 1.0)
    alpha = np.clip(alpha, 0.5, 1.0)
    
    # Base score: EI adjusted for cost
    score = ei / (cost_safe ** alpha)
    
    # --- Add exploration bonus for high-uncertainty regions ---
    # When sigma is high, there's potential for finding better solutions
    # This is already partially captured by EI, but we add a small bonus
    # to encourage exploring truly uncertain regions
    
    # Normalize sigma for the bonus
    sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min + 1e-9)
    exploration_bonus = 0.15 * beta * sigma_norm / cost_safe
    
    score = score + exploration_bonus
    
    # --- Exploitation boost for points very close to f_best ---
    # If mu is very close to f_best (within a small tolerance), boost the score
    # This helps converge to the optimum
    
    mu_diff = f_best - mu  # Positive if mu < f_best (good)
    # Boost for points where mu is close to or below f_best
    close_mask = mu_diff > 0.05 * (sigma_mean + 1e-9)
    if np.any(close_mask):
        # Additional boost proportional to how much below f_best mu is
        boost = 0.3 * np.maximum(mu_diff, 0) / (sigma_mean + 1e-9)
        score[close_mask] += boost[close_mask]
    
    # --- Final numerical cleanup ---
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    score = np.maximum(score, 0.0)  # Ensure non-negative
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > DEADLINE:
        # Return immediately with whatever we have
        pass
    
    return score
