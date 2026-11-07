"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline enforcement
_PROGRAM_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    # 1. Deadline Check: return a fast, valid score if we're running out of time
    elapsed = time.time() - _PROGRAM_START_TIME
    if elapsed > _TIME_LIMIT:
        # Fast fallback: simple EI / cost
        safe_sigma = sigma + 1e-9
        z = (f_best - mu) / safe_sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / (cost + 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # 2. Adaptive Acquisition: Blend EI and UCB
    # Early on: more exploration (UCB weight higher)
    # Later on: more exploitation (EI weight higher)
    
    # Heuristic for adaptivity:
    # As f_best decreases, we want to exploit more.
    # f_best starts high (Rosenbrock max is ~20 in this domain? No, range is [-2,2]^2.
    # f(2,2) = (1-4)^2 + 16*4 = 9+64=73. f(0,0)=0.
    # So f_best goes from ~73 down to 0.
    
    # Let's use a normalized progress metric.
    # If f_best is close to 0, we are in exploitation mode.
    # If f_best is large, we are in exploration mode.
    
    # Simple adaptive beta for UCB part:
    # beta = 1.0 + 1.0 * (f_best / 10.0) -> capped?
    # Or better: blend the two scores directly.
    
    # Calculate EI
    safe_sigma = sigma + 1e-9
    z = (f_best - mu) / safe_sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_cost = ei / cost
    
    # Calculate UCB (minimization: -mu + beta*sigma)
    # Adaptive beta: start high, decrease as f_best approaches 0
    # f_best range is roughly [0, 73].
    # Let's map f_best to a weight for exploration.
    # weight_explore = 1.0 if f_best > 10, 0.0 if f_best < 1.
    # Linear interpolation in between.
    
    f_best_val = float(f_best)
    # Clamp f_best for the heuristic
    f_for_weight = np.clip(f_best_val, 0.0, 10.0)
    
    # When f_best is 10, weight_explore = 1.0 (high exploration)
    # When f_best is 0, weight_explore = 0.0 (pure exploitation)
    weight_explore = f_for_weight / 10.0
    
    # UCB beta: scale with weight_explore?
    # If weight_explore is high, we want high beta.
    # If weight_explore is low, we want low beta (or just use EI).
    
    # Let's try a blend:
    # Score = (1 - weight_explore) * EI/Cost + weight_explore * UCB/Cost
    # But UCB needs a beta. Let's fix beta=2.0 for the UCB part.
    beta = 2.0
    ucb = -mu + beta * sigma
    ucb_cost = ucb / cost
    
    # Normalize the two scores to have similar magnitudes?
    # EI is always positive. UCB can be negative.
    # If we just add them, UCB might dominate if mu is large positive.
    # But mu is in objective units.
    
    # Alternative: Adaptive Beta in a single UCB-like formula?
    # Score = -mu + beta(f_best) * sigma
    # beta(f_best) = 0.5 + 1.5 * (f_best / 10)
    # If f_best=0, beta=0.5 (exploitation)
    # If f_best=10, beta=2.0 (exploration)
    
    beta_adaptive = 0.5 + 1.5 * (f_for_weight / 10.0)
    
    ucb_adaptive = -mu + beta_adaptive * sigma
    scores = ucb_adaptive / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
