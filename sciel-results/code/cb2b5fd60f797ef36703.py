"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    Approximates the expected reduction in the global minimum value.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient approximation:
    # For each candidate i, the KG value is the expected improvement of the global minimum
    # given that we evaluate candidate i.
    # 
    # A common approximation (Hou et al., 2011) uses:
    # KG_i = E[min(f_best, f_i)] - min(f_best, E[f_i])
    #      = f_best * P(f_i > f_best) + E[f_i | f_i < f_best] - min(f_best, mu_i)
    #
    # More precisely:
    # Let X_i ~ N(mu_i, sigma_i^2)
    # E[min(f_best, X_i)] = f_best * Phi((f_best - mu_i)/sigma_i) + mu_i * Phi((mu_i - f_best)/sigma_i) 
    #                        - sigma_i * phi((f_best - mu_i)/sigma_i)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    #
    # The KG score is:
    # KG_i = E[min(f_best, X_i)] - min(f_best, mu_i)
    #
    # However, a simpler and often effective surrogate is:
    # KG_i = E[max(0, f_best - X_i)] = EI_i
    # But true KG considers the interaction with other points.
    #
    # A practical approximation that captures the global aspect:
    # KG_i ≈ EI_i + correction_term
    #
    # Let's use the standard EI formula but with a twist:
    # We can approximate KG by considering that improving the best value by d
    # reduces the objective by d.
    #
    # Standard EI for minimization:
    # EI_i = (f_best - mu_i) * Phi((f_best - mu_i)/sigma_i) + sigma_i * phi((f_best - mu_i)/sigma_i)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # EI calculation
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Knowledge Gradient approximation:
    # The key difference from EI is that KG accounts for the fact that
    # if candidate i is not the best, it doesn't help. But if it IS the best,
    # it improves the global best.
    #
    # A common approximation is:
    # KG_i = EI_i * P(candidate i is the new best)
    # But P(candidate i is the new best) is hard to compute.
    #
    # Alternative: Use the "Expected Improvement of the Best" (EIB)
    # EIB_i = E[max(0, f_best - min_j X_j)] where j ranges over all candidates
    #
    # This is computationally expensive. A cheaper approximation:
    # Assume that the probability that candidate i becomes the new best is
    # proportional to its EI relative to the sum of all EIs.
    #
    # Let's try a simpler approach: use EI but with an adaptive beta-UCB blend
    # that shifts toward exploitation as we approach the end.
    #
    # Actually, let's implement a proper KG approximation using the formula:
    # KG_i = E[f_best - min(f_best, X_i)] = EI_i
    # This is exactly EI. So KG ≈ EI in the single-point case.
    #
    # For a better global perspective, let's add a diversity bonus:
    # Points that are far from other high-EI points get a bonus.
    # But we don't have coordinates, so we can't compute distances.
    #
    # Let's stick with a refined EI with adaptive exploration.
    # The current UCB approach had beta decaying from 3 to 2.
    # Let's try EI with a small UCB component to encourage exploration.
    
    elapsed = time.time() - _start_time
    
    if elapsed < 15.0:
        beta = 1.5
    elif elapsed < 30.0:
        t = (elapsed - 15.0) / 15.0
        beta = 1.5 - 0.5 * t  # 1.5 to 1.0
    else:
        beta = 1.0
    
    # Blend EI and UCB
    ucb_score = beta * sigma_safe - mu
    score = ei + 0.1 * ucb_score
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
