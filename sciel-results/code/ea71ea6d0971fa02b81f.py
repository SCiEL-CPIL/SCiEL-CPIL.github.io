import numpy as np
import time

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Pareto-frontier guided exploration
    # Identify the exploitation anchor: candidate with lowest predicted mean
    mu_min = np.min(mu)
    
    # Compute normalized distance from anchor in prediction space
    # For points close to the anchor in mu, we want high scores if sigma is high
    # For points far from anchor, we want lower scores
    
    # Normalize mu distance: 0 at anchor, 1 at max
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-8:
        dist_norm = (mu - mu_min) / (mu_max - mu_min)
    else:
        dist_norm = np.zeros_like(mu)
    
    # The score combines:
    # 1. Exploitation: prefer low mu (high -mu)
    # 2. Exploration: prefer high sigma, but modulated by distance from anchor
    #    Near the anchor, we want to explore (high sigma gets boosted)
    #    Far from anchor, we want to exploit (high mu gets penalized)
    
    # Geometric approach: score = sigma * (1 - dist_norm) + exploitation_term
    # This prioritizes high-uncertainty regions near the best-predicted area
    
    # Adaptive balance based on progress
    # When f_best is large (early), explore more
    # When f_best is small (late), exploit more
    progress = f_best / (f_best + 1.0)
    
    # Exploration weight: high early, low late
    explore_weight = 1.0 + 2.0 * progress
    
    # Exploitation weight: always present
    exploit_weight = 1.0
    
    # Score: combine exploitation and exploration
    # Exploitation: -mu (lower is better, so we maximize -mu)
    # Exploration: sigma * (1 - dist_norm) * explore_weight
    scores = exploit_weight * (-mu) + explore_weight * sigma * (1 - dist_norm)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
