"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-Adaptive Expected Improvement.
    
    Instead of a fixed xi, we estimate a dynamic xi based on the distribution
    of potential improvements across the candidate set. This allows the 
    acquisition function to balance exploration and exploitation more 
    effectively than standard EI with xi=0.
    """
    # Ensure inputs are numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal variable
    z = (f_best - mu) / sigma_safe
    
    # Standard EI with xi = 0
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_standard = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate a "batch temperature" or dynamic xi.
    # We look at the top 10% of EI scores to see how much improvement is "easy" to find.
    # If the top EI values are small, it suggests we are in a region where the GP is 
    # confident or the improvement is marginal, so we might need to explore more (higher xi).
    # However, a simpler heuristic often used is to scale xi based on the standard deviation
    # of the improvements or the ratio of uncertainty to improvement.
    
    # Let's try a different structural approach: 
    # Use the median sigma to scale xi. 
    # If sigma is high everywhere, the GP is uncertain, so we can afford a lower xi (more exploitation 
    # of the mean) because the mean itself is noisy. 
    # If sigma is low, the GP is confident, so we need a higher xi to force exploration away from the known best.
    
    # Actually, the standard intuition is:
    # High uncertainty -> Explore (High sigma regions). EI already does this via the sigma * pdf(z) term.
    # Low uncertainty -> Exploit (Low mu regions).
    
    # The "Batch-Adaptive" idea: 
    # Let's estimate the expected improvement of the *best* candidate in the set.
    # If this value is very small, we are stuck. We should increase exploration.
    
    # Simple adaptive xi:
    # xi = 0.01 * (f_best - min(mu)) / (mean(sigma) + 1e-9)
    # This scales xi with the potential range of improvement relative to uncertainty.
    
    min_mu = np.min(mu)
    mean_sigma = np.mean(sigma_safe)
    
    # Potential improvement range
    potential_range = f_best - min_mu
    
    # If potential_range is small compared to uncertainty, we are in a "plateau" of the surrogate.
    # We want to explore more.
    # Let's set xi proportional to the uncertainty-to-improvement ratio.
    # xi = 0.1 * mean_sigma / (potential_range + 1e-9)
    
    # Cap xi to prevent excessive exploration
    xi = np.clip(0.1 * mean_sigma / (potential_range + 1e-9), 0.0, 0.5)
    
    # Recalculate EI with adaptive xi
    # EI(xi) = (f_best - xi - mu) * Phi(z') + sigma * phi(z')
    # where z' = (f_best - xi - mu) / sigma
    z_prime = (f_best - xi - mu) / sigma_safe
    
    # Ensure z_prime is finite
    z_prime = np.nan_to_num(z_prime, nan=0.0, posinf=10.0, neginf=-10.0)
    
    ei_adaptive = (f_best - xi - mu) * norm.cdf(z_prime) + sigma_safe * norm.pdf(z_prime)
    
    # Ensure scores are finite
    scores = np.nan_to_num(ei_adaptive, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
