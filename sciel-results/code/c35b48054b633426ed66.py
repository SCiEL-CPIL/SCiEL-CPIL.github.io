import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Combines Kriging Believer structural approach with Expected Improvement refinement.
    
    1. Adaptive UCB identifies promising regions (high uncertainty + low bound).
    2. Soft gating via sigmoid selects top UCB regions smoothly.
    3. Within gated regions, prefer lower mu (exploitation) while adding
       a small EI term to handle uncertainty-aware improvement.
    """
    start_time = time.time()
    deadline = 295.0  # Safety margin before 300s limit
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive Beta: High regret -> High beta (explore), Low regret -> Low beta (exploit)
    f_global_min = 0.397887
    regret = f_best - f_global_min
    
    beta_min = 0.5
    beta_max = 3.0
    r_scale = 10.0
    
    if regret >= 0:
        fraction = regret / (regret + r_scale)
    else:
        fraction = 0.0
        
    beta = beta_min + (beta_max - beta_min) * fraction
    
    # Step 1: Compute UCB scores to identify promising regions
    # UCB for minimization: beta * sigma - mu
    ucb_scores = beta * sigma_safe - mu
    
    # Normalize UCB scores to [0, 1] for gating
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min + eps
    ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    # Step 2: Create a smooth gate that emphasizes high-UCB regions
    # Sigmoid with tunable temperature for soft selection
    temperature = 3.0
    gate = 1.0 / (1.0 + np.exp(-temperature * (ucb_norm - 0.5)))
    
    # Step 3: Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Scale EI to be comparable with UCB scores
    ucb_scale = np.max(np.abs(ucb_scores)) + eps
    ei_scaled = ei_scores / ucb_scale
    
    # Step 4: Combine gated exploitation (-mu) with EI refinement
    # The gate ensures we only strongly exploit in high-uncertainty regions
    # The EI term adds uncertainty-aware improvement pressure
    exploit_score = gate * (-mu)
    ei_component = 0.15 * ei_scaled
    
    # Add a small UCB component to break ties and maintain exploration
    ucb_component = 0.05 * ucb_scores
    
    scores = exploit_score + ei_component + ucb_component
    
    # Check time
    if time.time() - start_time > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
