"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 330.0

def _elapsed():
    return time.time() - _T0

def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    # sigma should be positive. GP noise is 1e-6, so sigma shouldn't be 0, but just in case.
    sigma = np.clip(sigma, 1e-6, None)
    cost = np.clip(cost, 1e-6, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Calculate Expected Improvement (for minimization)
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values that might cause numerical issues in cdf/pdf
    # although scipy is robust, clipping to [-10, 10] is safe.
    z = np.clip(z, -10.0, 10.0)
    
    # Compute EI
    # Using standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware EI score
    ei_score = ei / cost
    
    # UCB term: mu - beta * sigma (for minimization, we want low mu and high uncertainty)
    # For minimization, UCB = mu - beta * sigma encourages exploring high-uncertainty regions
    # while also preferring low mean predictions
    beta = 0.5
    ucb = (mu - beta * sigma) / cost
    
    # Bounded exploration bonus using soft-max style: sigma / (sigma + epsilon)
    # This is smoother and more differentiable than hard bounds
    epsilon = 0.05
    exploration = (sigma / (sigma + epsilon)) / cost
    
    # Weighted combination: primary weight on EI, secondary on UCB and exploration
    # EI is the main driver, UCB adds exploration pressure, exploration bonus is subtle
    w_ei = 1.0
    w_ucb = 0.1
    w_explore = 0.03
    
    score = w_ei * ei_score + w_ucb * ucb + w_explore * exploration
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
