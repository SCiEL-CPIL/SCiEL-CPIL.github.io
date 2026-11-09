"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 250.0  # safety margin before 300s limit
    
    # Ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Adaptive exploration parameter:
    # Early in optimization: more exploration; later: more exploitation
    # We use f_best to gauge progress: as f_best gets closer to 0, reduce beta
    # For Ackley, minimum is 0, so f_best decreasing toward 0 means progress
    
    # Estimate progress: if f_best is close to 0, we're near the optimum
    # Use a smooth schedule
    # Initially f_best might be ~1-10, goal is 0
    # beta_high = 4.0 (early, lots of exploration)
    # beta_low = 0.5 (late, exploitation-focused)
    
    # Use a normalized progress measure
    # Clamp f_best for numerical stability
    f_best_clamped = np.clip(f_best, 1e-10, None)
    
    # Progress: 1 when f_best is large (early), 0 when f_best ~ 0 (late)
    # Use log scale: log(f_best) goes from ~2-10 down to -inf
    # Normalize: progress = max(0, min(1, 1 - log(f_best+eps)/log(f_max+eps)))
    # But we don't know f_max. Use a simpler heuristic:
    # beta = beta_low + (beta_high - beta_low) * exp(-f_best / scale)
    # When f_best is large (early), exp(-large/scale) ~ 0, so beta ~ beta_low? 
    # Wait, we want high beta early. Let me rethink.
    
    # Actually: early on, f_best is high (bad), we want high beta (exploration)
    # Late on, f_best is low (good), we want low beta (exploitation)
    # So beta should be high when f_best is high, low when f_best is low
    # beta = beta_low + (beta_high - beta_low) * (f_best / (f_best + f_scale))
    
    f_scale = 0.5  # Ackley values near origin are small; 0.5 is a reasonable scale
    beta = 0.3 + 3.7 * (f_best / (f_best + f_scale))
    # When f_best >> f_scale: beta ~ 4.0
    # When f_best << f_scale: beta ~ 0.3
    
    # Also add a time-based component: early calls (few evaluations) -> more exploration
    # But we don't know the step number. Use mu statistics as a proxy.
    # If mu has high variance or sigma is high relative to mu, we're still exploring
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # But we want a combination that's more aggressive
    
    # Let's use a blend of EI and UCB
    # EI-based score:
    eps = 1e-10
    z = (f_best - mu) / (sigma + eps)
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # UCB score:
    ucb = -mu + beta * sigma
    
    # Blend: use EI as primary, UCB as secondary
    # Normalize both to have similar scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_norm = (ei - ei_min) / (ei_max - ei_min + eps)
    ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min + eps)
    
    # Blend factor: early -> more UCB (exploration), late -> more EI (exploitation)
    # Use same beta-like schedule
    blend_ucb = f_best / (f_best + f_scale)  # 1 early, 0 late
    scores = blend_ucb * ucb_norm + (1 - blend_ucb) * ei_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() > deadline:
        # Fallback to simple UCB
        sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), 0.0)
        scores = -mu + 1.5 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
