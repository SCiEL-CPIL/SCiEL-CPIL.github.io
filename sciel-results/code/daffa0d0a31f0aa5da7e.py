"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive expected improvement per unit cost.
    
    Strategy:
    1. Compute standard Expected Improvement (EI).
    2. Divide by cost^alpha, where alpha is adaptive based on the current state of exploration.
       - Early on (high mean sigma), use a lower alpha (e.g., ~0.5) to allow more exploration
         even at higher costs, as we need to map the function landscape.
       - Late on (low mean sigma), use a higher alpha (e.g., ~1.2) to strictly conserve budget
         and exploit cheap, high-improvement points.
    3. Add a small, adaptive exploration bonus to prevent collapse to a single point when EI is low.
    4. Include a time guard to ensure we return within the 300s limit (though this function is fast).
    """
    # Time guard: track start time
    start_time = time.time()
    deadline = 290.0  # Safety margin
    
    # Ensure inputs are numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement calculation
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z to avoid numerical overflow/underflow in cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    ei = delta * cdf_val + sigma_safe * pdf_val
    # EI should be non-negative by definition, but numerical errors can cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost exponent
    # Measure the current level of exploration (uncertainty)
    # If mean sigma is high, we are in the exploration phase -> lower alpha (less penalty for cost)
    # If mean sigma is low, we are in the exploitation phase -> higher alpha (strong penalty for cost)
    
    mean_sigma = np.mean(sigma_safe)
    
    # Normalize mean_sigma to a range that makes sense for adaptation
    # Typical sigma values in standardized objectives might be 0.1 - 1.0
    # Let's map mean_sigma to alpha in [0.5, 1.2]
    # If mean_sigma is very high (e.g., > 1.0), alpha = 0.5
    # If mean_sigma is very low (e.g., < 0.1), alpha = 1.2
    
    # Simple linear mapping with clipping
    # alpha = 1.2 - 0.7 * (mean_sigma / 1.0)  # This would make high sigma -> low alpha
    # But we want to cap it.
    
    # Let's use a smoother transition.
    # alpha = 0.5 + 0.7 * (1 - min(mean_sigma, 1.0))
    # If mean_sigma = 1.0, alpha = 0.5
    # If mean_sigma = 0.0, alpha = 1.2
    # If mean_sigma = 0.5, alpha = 0.85
    
    normalized_sigma = np.clip(mean_sigma, 0.0, 1.0)
    alpha = 0.5 + 0.7 * (1.0 - normalized_sigma)
    
    # Compute cost-aware EI
    # score_ei = ei / (cost_safe ** alpha)
    
    # Check time
    if time.time() > deadline - 5.0:
         # Fallback: simple EI / cost
         score = ei / cost_safe
         score = np.where(np.isfinite(score), score, 0.0)
         return score
    
    cost_factor = cost_safe ** alpha
    score_ei = ei / cost_factor
    
    # Exploration bonus
    # When EI is low (mu >= f_best), we still want to explore high sigma regions.
    # The bonus should also be cost-aware.
    # bonus = sigma_safe / (cost_safe ** alpha)
    # Weight the bonus lightly.
    # If alpha is low (exploration phase), the bonus is less penalized by cost.
    # If alpha is high (exploitation phase), the bonus is heavily penalized, focusing on EI.
    
    # Let's use a small weight, e.g., 0.05, to keep EI dominant but provide a floor.
    exploration_bonus = 0.05 * (sigma_safe / cost_factor)
    
    score = score_ei + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
