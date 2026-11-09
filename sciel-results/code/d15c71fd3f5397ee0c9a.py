"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and Uncertainty (UCB-like).
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
       EI(x) = E[max(f_best - f(x), 0)]
       For f(x) ~ N(mu, sigma^2), this is:
       (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
       where Phi is the CDF and phi is the PDF of the standard normal distribution.
       
    2. Add an exploration bonus based on sigma.
       This encourages visiting regions with high uncertainty, which is crucial for 
       escaping local minima in multimodal functions like Weierstrass, especially 
       in the early stages when the GP posterior is uncertain.
       
    3. The total score is EI + kappa * sigma.
       Since we want higher scores to be selected, and EI is higher when potential 
       improvement is higher, this aligns with the requirement.
       
    We use a fixed kappa (e.g., 0.1) to balance exploitation (EI) and exploration (sigma).
    The value of kappa might need tuning, but 0.1 is a reasonable starting point for 
    standardized objectives where sigma is in objective units.
    
    Note: The problem states mu and sigma are in objective units.
    f_best is the lowest observed value.
    
    Edge cases:
    - If sigma is very small, EI approaches max(f_best - mu, 0).
    - If sigma is large, the EI term behaves like a smoothed version of the potential improvement,
      and the exploration term dominates, pushing the search towards unexplored areas.
    """
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero in the z-score calculation
    # We add a small epsilon to sigma to prevent numerical issues
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the z-score: z = (f_best - mu) / sigma
    # This represents how many standard deviations f_best is above the mean.
    # If f_best > mu, z is positive, meaning we expect improvement.
    z = (f_best - mu) / sigma_safe
    
    # Calculate the Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    # For numerical stability and efficiency, we can use scipy.stats.norm
    # norm.cdf(z) and norm.pdf(z)
    
    # However, for very large |z|, norm.cdf and norm.pdf can underflow or behave poorly.
    # norm.cdf(z) -> 1 for large z, 0 for large -z.
    # norm.pdf(z) -> 0 for large |z|.
    
    # Let's compute the components
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure EI is non-negative (numerical artifacts can sometimes make it slightly negative)
    ei = np.maximum(ei, 0)
    
    # Exploration bonus
    # We add a term proportional to sigma.
    # The scale of this term relative to EI depends on the magnitude of the objective function.
    # Since the Weierstrass function is bounded and standardized targets are mentioned 
    # (though mu/sigma are in objective units, the GP standardizes targets internally, 
    # but the output is transformed back), the scale of sigma is relevant.
    
    # A common choice is to use a small multiple of sigma.
    # Let's try kappa = 0.1.
    # This adds a gentle push towards high uncertainty.
    kappa = 0.1
    exploration_bonus = kappa * sigma
    
    # Total score
    score = ei + exploration_bonus
    
    # Ensure finite scores
    # If any non-finite values exist, replace them with a very low score (0 or -inf)
    # so they are not selected.
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
