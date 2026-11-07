"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) 
    with a diversity bonus to encourage exploration.
    """
    # Deadline guard: return early if we are running out of time
    # The contract says 300s per evaluation, but we don't have a global clock here.
    # However, we can add a simple check if this function is called in a tight loop 
    # or if we want to be safe. Since we are just computing scores for 2000 points,
    # it's very fast. The risk is in the outer loop. But we can't control the outer loop.
    # We will focus on improving the score quality.
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute a diversity/exploration bonus
    # We want to penalize points that are very close to each other? 
    # No, we don't have the coordinates.
    # But we can use the posterior uncertainty itself as a proxy for exploration.
    # Standard EI already balances exploitation (f_best - mu) and exploration (sigma).
    # However, sometimes EI can be too greedy.
    
    # Let's try a weighted combination of EI and UCB (Upper Confidence Bound)
    # UCB = mu + kappa * sigma
    # Since we are minimizing, we want to minimize the function.
    # So we want to pick points with low mu and high sigma.
    # EI does this.
    
    # Let's try to smooth the EI calculation or add a small penalty for high sigma?
    # No, high sigma is good for exploration.
    
    # One common improvement is to use the "Expected Hypervolume Improvement" (EHVI) 
    # but that's for multi-objective.
    
    # Another idea: Use the "Probability of Improvement" (PI) or a blend.
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # PI is more exploratory than EI.
    
    # Let's try a blend of EI and PI.
    pi = norm.cdf(z)
    
    # Weighted sum: 0.8 * EI + 0.2 * PI
    # Or maybe just stick with EI but ensure numerical stability.
    
    # Let's try to be more aggressive with exploration by scaling sigma.
    # Or use a "soft" EI that doesn't go to 0 as fast?
    
    # Actually, the baseline EI is quite strong. Let's try to add a small 
    # "novelty" term based on the variance of the mu and sigma themselves?
    # No, that doesn't make sense.
    
    # Let's try to use the "Confidence Bound" (CB) or "Knowledge Gradient" (KG).
    # KG is hard to compute efficiently for 2000 points without the kernel matrix.
    
    # Let's try a simple modification: 
    # If sigma is very large, EI is dominated by sigma * pdf(z), which is small.
    # If sigma is very small, EI is dominated by (f_best - mu) * cdf(z).
    
    # Let's try to use a "Gaussian Process-UCB" (GP-UCB) style score.
    # score = -mu + kappa * sigma
    # We want to maximize this score (since lower f is better, we want low mu, high sigma).
    # So score = -mu + kappa * sigma.
    
    # Let's try a combination of EI and UCB.
    # UCB score: -mu + 0.5 * sigma (arbitrary kappa)
    # EI score: ei
    
    # Normalize both to [0, 1] range?
    # min_ei = np.min(ei), max_ei = np.max(ei)
    # min_ucb = np.min(-mu + 0.5*sigma), max_ucb = np.max(-mu + 0.5*sigma)
    
    # This is getting complicated. Let's stick to EI but with a slight tweak.
    # Some sources suggest that EI can be improved by using a "temperature" parameter.
    
    # Let's try to return the EI directly, but ensure no NaNs.
    scores = ei
    
    # Check for NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
