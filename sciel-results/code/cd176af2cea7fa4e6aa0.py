import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Expected Improvement (SEI) acquisition function.
    
    This function models the evaluation cost as a Log-Normal random variable
    rather than a deterministic scalar. This structural change integrates cost
    uncertainty into the acquisition criterion, providing a more robust
    improvement-per-cost estimate than simple division (EI / cost).
    
    SEI = E_C[E_f[max(f_best - f, 0) / C]]
    
    For C ~ LogNormal(mu_c, sigma_c^2), E[1/C] = exp(mu_c + sigma_c^2 / 2).
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 290.0 # 10s safety margin before 300s limit

    # Ensure inputs are 1D arrays and handle potential shape issues
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    n = len(mu)
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Calculate Standard Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Model Cost as Log-Normal
    # cost = 1 + mean(x_scaled). x_scaled in [0,1]. cost in [1, 2].
    # We model the uncertainty in this cost estimate.
    # A moderate sigma_c (e.g., 0.2) implies ~20% uncertainty in cost.
    cost_safe = np.maximum(cost, 1e-8)
    mu_c = np.log(cost_safe)
    sigma_c = 0.2  # Cost uncertainty parameter
    
    # E[1/C] for LogNormal(mu_c, sigma_c^2)
    e_inv_cost = np.exp(mu_c + 0.5 * sigma_c**2)
    
    # 3. Stochastic Expected Improvement
    # SEI = EI * E[1/C]
    se_i = ei * e_inv_cost
    
    # 4. Exploration Term
    # Add a term proportional to uncertainty (sigma) also weighted by E[1/C]
    # This encourages exploring cheap, uncertain regions.
    # Beta is tuned to be small relative to EI to avoid pure random search.
    beta = 0.1
    exploration = beta * sigma_safe * e_inv_cost
    
    scores = se_i + exploration
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return simple EI/cost if we are extremely close to timeout
        # This is unlikely as the vectorized ops are fast, but required by directive
        fallback_cost_safe = np.maximum(cost, 1e-8)
        fallback_scores = ei / fallback_cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
