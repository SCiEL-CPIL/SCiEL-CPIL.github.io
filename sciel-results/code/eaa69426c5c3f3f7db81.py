import numpy as np
import time

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard: return safe fallback if close to 330s limit
    if time.time() - _START_TIME > 325:
        # Fallback: simple UCB to ensure finite output
        return -mu + sigma
    
    n = len(mu)
    
    # Find the anchor: candidate with the lowest predicted mean (most promising exploitation point)
    anchor_idx = np.argmin(mu)
    mu_anchor = mu[anchor_idx]
    
    # Compute distance from anchor in prediction space
    # Use normalized distance to make it scale-invariant
    mu_spread = np.ptp(mu)
    if mu_spread > 1e-10:
        dist_from_anchor = (mu - mu_anchor) / mu_spread
    else:
        dist_from_anchor = np.zeros_like(mu)
    
    # The distance should be non-negative; candidates with mu > mu_anchor are "farther" in the exploitation direction
    # We want to explore regions that are uncertain AND potentially better than the anchor
    # But for minimization, we want low mu. So the "distance" that matters is how much worse the prediction is
    # Actually, we want to find points that are uncertain and where the prediction could be lower
    # Let's think differently: we want to score high where sigma is high AND mu is low relative to the landscape
    
    # Alternative: score = sigma * (1 - normalized_mu) where normalized_mu is high for good (low) predictions
    # Let's use: score = sigma * (f_best - mu) / (f_best + 1) + sigma * exploration_bonus
    
    # Actually, let's follow the directive more closely:
    # Score = sigma * normalized_distance_from_anchor
    # Where normalized_distance captures how "different" this candidate is from the best prediction
    # For minimization, we want candidates where mu is low (good) and sigma is high (uncertain)
    
    # Let's use a hybrid: the product of sigma and a measure of "potential improvement"
    # potential_improvement = max(0, f_best - mu) / (f_best + 1e-8)
    # But this could be zero for most points if f_best is very low
    
    # Following the directive exactly:
    # 1. Anchor = argmin(mu)
    # 2. Score = sigma * normalized_distance_from_anchor_in_prediction_space
    
    # Distance from anchor: how far is this candidate's mu from the anchor's mu
    # Normalize by the spread of mu to make it scale-invariant
    if mu_spread > 1e-10:
        norm_dist = np.abs(mu - mu_anchor) / mu_spread
    else:
        norm_dist = np.zeros_like(mu)
    
    # But we also want to favor candidates where mu is low (better for minimization)
    # So let's weight the distance by how much lower mu is than f_best
    # Or simply: score = sigma * (1 + norm_dist) * (1 - mu_normalized)
    
    # Let's try: score = sigma * (1 + norm_dist)
    # This rewards high uncertainty and being far from the anchor in prediction space
    # The "far" could mean either better or worse predictions, but combined with high sigma,
    # it encourages exploring uncertain regions that are different from the current best guess
    
    # To bias toward exploitation (low mu), let's add a small exploitation term
    # or use: score = sigma * (1 + norm_dist) - 0.1 * mu_normalized
    
    # Actually, let's keep it simple and follow the directive:
    # score = sigma * normalized_distance
    # where normalized_distance = |mu - mu_anchor| / mu_spread
    
    # But this symmetric distance doesn't prefer lower mu. Let's make it asymmetric:
    # distance = (mu_anchor - mu) / mu_spread for candidates better than anchor (negative = better)
    # and (mu - mu_anchor) / mu_spread for candidates worse
    # We want to explore candidates that are uncertain and potentially better
    
    # Let's use: score = sigma * (1 + max(0, (mu_anchor - mu) / mu_spread))
    # This gives a boost to candidates that are both uncertain and predicted to be better than the anchor
    
    if mu_spread > 1e-10:
        # Asymmetric distance: positive for candidates better than anchor
        improvement_potential = np.maximum(0, (mu_anchor - mu) / mu_spread)
        # Also consider candidates that are worse but very uncertain (could be lucky)
        # distance = 1 + improvement_potential + 0.5 * np.maximum(0, (mu - mu_anchor) / mu_spread)
        distance = 1.0 + improvement_potential
    else:
        distance = np.ones_like(mu)
    
    scores = sigma * distance
    
    # Add a small exploitation component to ensure we don't completely ignore low-mu regions
    # This helps when sigma is small but mu is very low
    if mu_spread > 1e-10:
        exploitation_boost = 0.5 * np.maximum(0, (f_best - mu) / (f_best + 1e-8))
    else:
        exploitation_boost = np.zeros_like(mu)
    
    scores = scores + exploitation_boost * np.max(sigma)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
