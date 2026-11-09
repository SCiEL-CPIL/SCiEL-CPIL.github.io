import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG).
    
    KG estimates the expected improvement in the global minimum by approximating
    the posterior predictive distribution at the top-k candidates. This approach
    directly optimizes for the reduction in expected regret of the best solution
    found, which is more robust for multimodal functions like Styblinski-Tang.
    
    Args:
        mu: Mean of the GP posterior at candidates (shape 2000,).
        sigma: Std dev of the GP posterior at candidates (shape 2000,).
        f_best: Current best function value (minimum).
        
    Returns:
        scores: Array of shape 2000, higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to EI if too close to deadline
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient approach:
    # 1. Identify top-k candidates most likely to improve the best
    # 2. For each candidate, estimate E[min(f_best, f(x))] using MC
    # 3. KG score = f_best - E[min(f_best, f(x))]
    
    # Select top-k candidates based on potential to beat f_best
    # Use a combination of low mean and high uncertainty
    improvement_potential = (f_best - mu) / (sigma_safe + 1e-9)
    top_k_indices = np.argsort(-improvement_potential)[:50]
    
    # Monte Carlo estimate of KG
    n_mc = 20  # Number of MC samples for KG estimation
    rng = np.random.default_rng()
    
    # Initialize KG scores
    kg_scores = np.zeros(n)
    
    # For each top-k candidate, compute KG via MC
    for idx in top_k_indices:
        mu_i = mu[idx]
        sigma_i = sigma_safe[idx]
        
        # Sample from posterior predictive at this candidate
        samples = rng.normal(mu_i, sigma_i, size=n_mc)
        
        # E[min(f_best, f(x))] = mean of min(f_best, samples)
        expected_min = np.mean(np.minimum(f_best, samples))
        
        # KG = f_best - E[min(f_best, f(x))]
        kg_scores[idx] = f_best - expected_min
    
    # For candidates not in top-k, use a heuristic based on EI
    # This provides exploration for candidates outside the top-k
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine: use KG for top-k, EI for others
    # Scale EI to be comparable to KG
    kg_max = np.max(kg_scores) if np.any(kg_scores > 0) else 1.0
    ei_max = np.max(ei_scores) if np.any(ei_scores > 0) else 1.0
    
    # Normalize and combine
    combined_scores = np.zeros(n)
    for i in range(n):
        if i in top_k_indices:
            combined_scores[i] = kg_scores[i]
        else:
            # Scale EI to match KG scale
            combined_scores[i] = ei_scores[i] * (kg_max / (ei_max + 1e-9)) * 0.5
    
    # Ensure all scores are finite
    scores = np.nan_to_num(combined_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
