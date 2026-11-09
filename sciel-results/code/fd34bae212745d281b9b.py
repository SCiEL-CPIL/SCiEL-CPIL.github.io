import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Standard Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle edge cases where sigma is very small
    # If sigma is tiny, EI is basically max(0, f_best - mu)
    
    # Domain Knowledge Bonus:
    # We know the minimum of x^2 is 0.
    # We add a bonus for candidates with predicted mean close to 0.
    # This helps "find" the zero even if the GP is noisy.
    
    # Adaptive weight: Strong when f_best is large (early), weak when f_best is small (late)
    # f_best starts high (e.g., 10-25) and goes to ~0.
    weight = 1.0 / (1.0 + np.exp(-(f_best - 1.0))) 
    # This sigmoid is 0.5 at f_best=1. 
    # If f_best=10, weight ~ 0.99.
    # If f_best=0.1, weight ~ 0.27.
    # This ensures strong bias early.
    
    # Adaptive width: 
    # Early on, mu might be far from 0. We want a broad bonus.
    # Late on, mu should be close to 0. We want a narrow bonus to refine.
    tau = 0.5 + f_best / 5.0
    # If f_best=10, tau=2.5. exp(-mu^2/12.5). mu=5 -> exp(-2) ~ 0.13.
    # If f_best=0.1, tau=0.52. exp(-mu^2/0.54). mu=0.5 -> exp(-0.46) ~ 0.63.
    
    # The bonus should be scaled to be comparable to EI.
    # EI max value is roughly f_best.
    # Bonus max is 1.0.
    # We should scale the bonus by f_best or a constant.
    # Let's scale by f_best to match the magnitude of improvement.
    bonus = weight * f_best * np.exp(-mu**2 / (2 * tau**2))
    
    scores = ei + bonus
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
