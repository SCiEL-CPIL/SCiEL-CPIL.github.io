"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Structural change: Cost-Aware Hybrid EI/UCB with adaptive exploration weight.
Reverts from Kriging Believer to an improvement-based metric which showed
better performance in the lineage (inspiration 1).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _fast_ei(mu, sigma, f_best):
    """
    Calculate Expected Improvement (EI) for minimization.
    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    where Z = (f_best - mu) / sigma
    """
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    z = (f_best - mu) / sigma_safe
    
    # Phi is CDF, phi is PDF
    # Use scipy.stats.norm for accuracy and speed
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    return ei

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Uses a hybrid of Expected Improvement (EI) and UCB, normalized by cost.
    The weight of exploration (UCB) adapts based on f_best.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate EI
    ei = _fast_ei(mu, sigma, f_best)
    
    # 2. Calculate UCB (Lower Confidence Bound for minimization, so we maximize -mu + beta*sigma)
    # We want to maximize the score.
    # UCB_score = -mu + beta * sigma
    # However, EI is in "improvement" units, UCB is in "function value" units.
    # To blend them, we can normalize or just use a weighted sum if scales are similar.
    # In Rastrigin, f_best starts high (~10) and goes to 0.
    # EI is roughly proportional to potential improvement.
    # A common approach is to use EI primarily and add a small UCB term for exploration.
    
    # Adaptive beta for UCB term
    # When f_best is high, we are far from optimum, need more exploration.
    # When f_best is low, we are close, need more exploitation.
    # Let's define exploration_weight w_e in [0, 1]
    # w_e = 1.0 when f_best is large, w_e = 0.0 when f_best is small.
    # Simple decay: w_e = 1 / (1 + f_best) ? No, f_best can be 0.
    # Let's use: w_e = min(1.0, max(0.0, 1.0 - f_best / 10.0))
    # If f_best = 0, w_e = 1.0 (max exploration? No, we want max exploitation).
    # Wait, if f_best is 0, we are at optimum. We should exploit (high EI, low UCB weight).
    # If f_best is 10, we are far. We should explore (high UCB weight).
    
    # Let's define:
    # score = w_e * UCB_term + (1 - w_e) * EI_term
    # But scales differ.
    # Alternative: score = EI + beta * sigma
    # This is a "Boosted EI".
    
    # Let's try a simpler structural change: Cost-Weighted EI with a UCB floor.
    # Or just stick to the Inspiration 1 which was a "fast adaptive blend".
    
    # Let's implement the blend:
    # alpha * EI + (1-alpha) * (-mu + beta*sigma)
    # This is tricky due to scales.
    
    # Better approach: Use EI as the primary driver, but adjust the "optimism"
    # of the GP prediction? No, we only have mu/sigma.
    
    # Let's use the "Cost-Aware EI" defined as:
    # Score = EI / cost
    # And add a small exploration bonus:
    # Bonus = beta * sigma / cost
    # Total Score = (EI + beta * sigma) / cost
    
    # Adaptive beta:
    # beta should be larger when we are uncertain or far from optimum.
    # Let's use beta = 1.0 + 2.0 * (1 - min(1.0, f_best/5.0))
    # If f_best = 0, beta = 3.0 (high exploration? No, if f_best=0 we are done).
    # Actually, if f_best is 0, EI is 0 everywhere. We should stop or do nothing.
    # But the loop continues.
    
    # Let's look at the Inspiration 1 description again: "fast, adaptive blend of EI and UCB".
    # Let's assume the successful blend was:
    # Score = EI + gamma * sigma
    # where gamma adapts.
    
    # Let's try:
    # base_score = EI
    # exploration_score = sigma
    # combined = base_score + beta * exploration_score
    
    # Adaptive beta:
    # Start with high beta, decrease as f_best decreases.
    # beta = max_beta * exp(-f_best / tau) + min_beta
    # If f_best is large, exp term ~ 1, beta ~ max_beta + min_beta.
    # If f_best is 0, exp term = 1, beta ~ max_beta + min_beta. This is wrong.
    # We want low beta when f_best is small.
    # beta = min_beta + (max_beta - min_beta) * (f_best / (f_best + tau))
    # If f_best -> 0, beta -> min_beta.
    # If f_best -> inf, beta -> max_beta.
    
    min_beta = 0.5
    max_beta = 2.0
    tau = 1.0
    
    beta = min_beta + (max_beta - min_beta) * (f_best / (f_best + tau))
    
    # Calculate combined score
    # Note: EI is already non-negative.
    # sigma is non-negative.
    # We want to maximize this.
    
    raw_score = ei + beta * sigma
    
    # Cost-aware normalization
    scores = raw_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle any remaining non-finite values
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
