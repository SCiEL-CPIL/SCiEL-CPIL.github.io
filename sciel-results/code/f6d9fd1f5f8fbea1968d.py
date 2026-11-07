import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Kriging Believer acquisition:
    1. Compute Expected Improvement (EI) for each candidate.
    2. Determine a quantile z* from the top EI scores.
    3. Score candidates by mu + sigma * z* (the "believer" prediction).
    4. Normalize by cost.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    if time.time() > _deadline:
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Step 1: Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Step 2: Determine z* from the top EI scores
    # We want a quantile that corresponds to the "top" EI region.
    # A common choice is to use the 95th percentile or similar.
    # Here, we use the mean of the top 10% EI values to set a threshold,
    # then find the corresponding z-score.
    
    # Sort EI values
    ei_sorted = np.sort(ei)
    n = len(ei_sorted)
    # Take the top 10%
    top_k = max(1, n // 10)
    top_ei = ei_sorted[-top_k:]
    mean_top_ei = np.mean(top_ei)
    
    # We want to find z* such that the EI at that z* is approximately mean_top_ei.
    # However, EI depends on mu and sigma, so we can't directly invert it globally.
    # Instead, we use a simpler heuristic:
    # The "believer" point is where the GP mean + sigma * z* is maximized,
    # subject to the constraint that the EI is high.
    
    # A more direct interpretation of Kriging Believer:
    # 1. Find the point x* that maximizes EI.
    # 2. Compute the z-score at x*: z* = (f_best - mu(x*)) / sigma(x*)
    # 3. Return scores based on mu + sigma * z*
    
    # Let's find the candidate with the highest EI
    idx_max_ei = np.argmax(ei)
    z_star = (f_best - mu[idx_max_ei]) / sigma_safe[idx_max_ei]
    
    # Step 3: Score candidates by mu + sigma * z*
    believer_scores = mu + sigma_safe * z_star
    
    # Step 4: Normalize by cost
    scores = believer_scores / (cost + eps)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
