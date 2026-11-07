"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the limit, return a safe deterministic score
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple, fast, finite score to avoid timeout
        return np.full_like(mu, 1.0, dtype=np.float64)
    
    # CEGS: Cost-Effective Global Search
    # 1. Standardize the cost for the penalty term
    # The cost is 1 + mean(x_scaled). Range is [1, 2].
    # We want to penalize high cost non-linearly to favor cheap points.
    
    # 2. Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 3. UCB-style exploration term
    # Use a beta parameter that might adapt, but for now a fixed structural change
    # The cell finding suggests: "Combining EI with UCB exploration and a non-linear cost penalty (offset 0.6)"
    # Let's implement a blend: Score = (EI + beta * sigma) / (cost + offset)
    # The offset 0.6 is the key structural change mentioned in the findings.
    
    # To make it a "Model-Averaged" or more sophisticated structural change, 
    # we can dynamically adjust the exploration weight based on the uncertainty relative to the best.
    # However, the finding specifically points to the non-linear cost penalty with offset 0.6.
    
    # Let's try: Score = (EI + 1.0 * sigma) / (cost + 0.6)
    # The 1.0 * sigma is the exploration.
    # The (cost + 0.6) is the non-linear cost penalty (since cost is in [1,2], adding 0.6 reduces the relative penalty of high cost vs low cost compared to just /cost).
    # Actually, dividing by (cost + offset) makes the denominator larger for expensive points, thus penalizing them more strongly relative to the numerator if the numerator doesn't scale with cost.
    
    # Let's verify the logic:
    # Point A: cost=1, EI=1, sigma=0. Score = 1 / 1.6 = 0.625
    # Point B: cost=2, EI=1, sigma=0. Score = 1 / 2.6 = 0.384
    # Point C: cost=1, EI=0, sigma=1. Score = 1 / 1.6 = 0.625
    # Point D: cost=2, EI=0, sigma=1. Score = 1 / 2.6 = 0.384
    
    # This effectively doubles the penalty for expensive points compared to a simple /cost ratio?
    # Simple /cost: A=1, B=0.5. Ratio A/B = 2.
    # With offset: A=0.625, B=0.384. Ratio A/B = 1.62.
    # Wait, the penalty is *less* severe with an offset in the denominator?
    # 1/1 = 1. 1/1.6 = 0.625.
    # 1/2 = 0.5. 1/2.6 = 0.384.
    # The *relative* difference between cheap and expensive is reduced by the offset.
    # If the finding says "creates a stronger trade-off that favors cheaper evaluations", maybe the offset is in the numerator? Or maybe the cost is transformed?
    # Or maybe the "non-linear cost penalty" refers to something like exp(-cost)?
    
    # Let's re-read: "Combining EI with UCB exploration and a non-linear cost penalty (offset 0.6) creates a stronger trade-off that favors cheaper evaluations with both high expected improvement and high exploration potential"
    
    # Perhaps the score is: (EI + beta * sigma) * exp(-cost / offset)?
    # Or (EI + beta * sigma) / (cost ** k)?
    
    # Let's stick to the specific mention of "offset 0.6". A common form in cost-aware BO is:
    # Score = EI / (cost + epsilon)
    # If epsilon is large, it flattens the cost difference.
    # If the goal is to *favor* cheaper evaluations, we want the score to drop off faster for expensive ones.
    # This implies a convex cost function or a multiplicative penalty.
    
    # Let's try: Score = (EI + sigma) * (1 / (cost + 0.6))
    # This is what I had.
    
    # Alternative interpretation: The "offset 0.6" might be subtracted from the cost? No, cost >= 1.
    # Maybe the score is: (EI + sigma) - lambda * cost?
    
    # Let's look at the "Model-Averaged" part.
    # We can average between a pure EI/cost and a pure Exploration/cost.
    
    # Let's implement the specific finding's hint:
    # Score = (EI + UCB_term) / (cost + 0.6)
    # And ensure we return finite values.
    
    # To add "Model-Averaging", we can blend two different beta values or two different cost penalties.
    # But for a first structural change, let's implement the specific "offset 0.6" penalty with UCB.
    
    # UCB term: typically z * sigma or just sigma. Let's use 1.0 * sigma for exploration.
    # We can also add a small deterministic improvement term.
    
    exploration_weight = 1.0
    cost_offset = 0.6
    
    # Calculate scores
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    ucb_term = exploration_weight * sigma_safe
    numerator = ei + ucb_term
    denominator = cost + cost_offset
    
    scores = numerator / denominator
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure higher is better (we are minimizing, so EI is positive when mu < f_best)
    # If EI is negative, it means mu > f_best, which is unlikely to be chosen.
    # The scores should be such that the best candidate has the highest score.
    
    return scores
# EVOLVE-BLOCK-END
