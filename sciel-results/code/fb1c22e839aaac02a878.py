import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier guided exploration acquisition function.
    
    Strategy:
    1. Identify the exploitation anchor (lowest predicted mean).
    2. Compute geometric distance from anchor in prediction space.
    3. Score = sigma * normalized_distance, prioritizing high-uncertainty
       regions far from the current best prediction.
    
    This structural change uses a product of uncertainty and geometric distance
    to create a novel exploration signal that naturally balances exploitation
    (near anchor) with exploration (far from anchor with high sigma).
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    if time.time() - _START_TIME > 320:
        return -mu  # Greedy fallback: minimize mu
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # --- Step 1: Identify exploitation anchor ---
    # Candidate with lowest predicted mean
    mu_anchor = np.min(mu)
    
    # --- Step 2: Compute geometric distance from anchor in prediction space ---
    # Distance in (mu, sigma) space, normalized by sigma scale
    mu_diff = mu - mu_anchor
    sigma_scale = np.median(sigma_safe[sigma_safe > 1e-10]) if np.any(sigma_safe > 1e-10) else 1.0
    sigma_scale = max(sigma_scale, 1e-10)
    
    # Geometric distance: combines mu deviation and sigma
    geo_distance = np.sqrt((mu_diff / sigma_scale)**2 + (sigma_safe / sigma_scale)**2)
    
    # Normalize geo_distance to [0, 1] range
    geo_max = np.max(geo_distance) if np.max(geo_distance) > 0 else 1.0
    geo_norm = geo_distance / geo_max
    
    # --- Step 3: Score = sigma * normalized_distance ---
    # Product of uncertainty and geometric distance
    # High sigma AND far from anchor gets highest score
    scores = sigma_safe * geo_norm
    
    # Add a tiny exploitation term to break ties and ensure monotonic behavior
    # When sigma is very small, prefer lower mu
    eps = 1e-6
    scores = scores + eps * (-mu)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
