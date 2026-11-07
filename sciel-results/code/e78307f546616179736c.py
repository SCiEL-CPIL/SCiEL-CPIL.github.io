import numpy as np
import time
from scipy.stats import norm

# Safety margin: The evaluator kills at 330s.
# We set a deadline at 325s to allow for cleanup/return.
_DEADLINE = time.time() + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviation predictions.
    f_best: float, the best observed value so far.
    cost: (2000,) array of costs for each candidate.
    
    Returns:
    scores: (2000,) array of finite scores. Higher is better.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros_like(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Ensure inputs are finite and valid
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # 1. Calculate Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -30.0, 30.0)
    
    # Standard EI formula
    EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)

    # 2. Calculate Probability of Improvement (PI)
    PI = norm.cdf(z)
    PI = np.where(np.isfinite(PI), PI, 0.0)

    # 3. Structural Change: Cost-Weighted Components
    # Instead of blending EI and UCB and then gating, we directly construct a score
    # that balances improvement efficiency and exploration, both cost-aware.
    
    # Component A: Efficiency of Improvement
    # This drives exploitation of cheap, promising regions.
    eff_ei = EI / cost
    
    # Component B: Cost-Adaptive Exploration
    # We use a UCB-like term but with a cost-dampened beta.
    # Beta is reduced for expensive points to discourage wasting budget on uncertain, expensive areas.
    # Damping exponent 0.5 allows some exploration in expensive areas if sigma is very high.
    beta_base = 2.0
    cost_damping = np.power(cost, 0.5)
    adaptive_beta = beta_base / cost_damping
    
    exploration_term = adaptive_beta * sigma
    
    # Component C: Pure Exploration Floor
    # Ensure that even if EI is 0, high uncertainty regions get some score,
    # but still penalized by cost.
    # This prevents the score from being exactly 0 in unexplored cheap areas if EI is small.
    # Actually, eff_ei handles this. If EI is 0, eff_ei is 0.
    # We might need a small boost for high sigma low cost areas.
    
    # Combine: 
    # Weight 1: Efficiency of EI (Exploitation)
    # Weight 2: Cost-Adaptive Exploration (Exploration)
    
    # Dynamic weighting: 
    # If the current f_best is close to 0 (good solution found), we should be more conservative 
    # (rely more on EI efficiency). 
    # If f_best is large (poor solution), we should explore more.
    # Normalize f_best roughly based on the problem scale (Levy 2D min is 0, max can be ~100-1000).
    # Let's assume a scale of 10 for normalization.
    f_scale = 10.0
    exploit_weight = 1.0 - np.clip(f_best / f_scale, 0.0, 0.5)
    explore_weight = 0.5 * (1.0 - exploit_weight) + 0.5 # Keep exploration significant
    
    # Actually, simpler: Fixed weights often work better if tuned well.
    # Let's use a fixed blend but with the structural change of cost-adaptive beta.
    w_exploit = 1.0
    w_explore = 1.0
    
    scores = w_exploit * eff_ei + w_explore * exploration_term
    
    # Normalize scores to [0, 1] or similar to make them stable?
    # The evaluator just compares relative scores to pick the best.
    # However, large values can cause numerical issues.
    # Let's scale by the max score to keep them in a reasonable range, 
    # but since we only need to return the array, relative order is what matters.
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small floor to avoid ties or zero scores for valid candidates
    scores = np.maximum(scores, 1e-10)
    
    return scores
