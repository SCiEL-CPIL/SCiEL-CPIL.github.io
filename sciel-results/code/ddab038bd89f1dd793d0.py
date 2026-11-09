"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Probability of Improvement (PoI) and
    Expected Improvement (EI), weighted by cost, with adaptive
    exploration based on uncertainty.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Probability of Improvement (PoI)
    poi = norm.cdf(z)
    
    # Cost-normalized scores
    ei_cost = ei / cost
    poi_cost = poi / cost
    
    # Adaptive mixing: weight EI more when we have high confidence in improvement,
    # weight PoI more when uncertainty is high relative to expected improvement
    # This helps balance exploitation and exploration
    
    # Measure relative uncertainty: high sigma relative to expected improvement
    # means we should explore more (higher PoI weight)
    # When EI is large relative to sigma, exploit more (higher EI weight)
    
    # Use a smooth blending based on the ratio
    # ratio = ei / sigma_safe, when ratio is large, EI dominates
    # when ratio is small, PoI dominates
    ratio = ei / (sigma_safe + 1e-8)
    
    # Smooth weight: sigmoid-like function of ratio
    # weight_ei in [0, 1], weight_poi = 1 - weight_ei
    weight_ei = 1.0 / (1.0 + np.exp(-ratio))  # sigmoid(ratio)
    weight_poi = 1.0 - weight_ei
    
    # Base score: adaptive mix of EI and PoI, cost-normalized
    score = weight_ei * ei_cost + weight_poi * poi_cost
    
    # Add a small uncertainty bonus to encourage exploration in high-variance regions
    # This helps escape local optima in multimodal functions like Shekel 4d
    # The bonus is scaled by cost to maintain cost-awareness
    sigma_mean = np.mean(sigma_safe)
    uncertainty_bonus = sigma_safe / (sigma_safe + sigma_mean + 1e-8)
    score += 0.05 * uncertainty_bonus / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
