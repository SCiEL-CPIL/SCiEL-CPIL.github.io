"""Cost-aware acquisition function for Bayesian optimization (powell 4d)"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = [None]

def acquisition(mu, sigma, f_best, cost):
    if _T0[0] is None:
        _T0[0] = time.time()
    
    elapsed = time.time() - _T0[0]
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Deadline guard: if we're close to the limit, just return a simple score
    if elapsed > 315.0:
        return sigma_safe / cost_safe
    
    # Infer number of evaluations from cost.
    # Initial 5 uniform points: each has cost ~ 1 + mean(x_scaled) where x_scaled is uniform on [0,1]
    # So mean cost per point ~ 1.5. Total initial cost ~ 7.5.
    # Each new evaluation adds ~1.5 cost.
    # So n_evals ≈ cost_total / 1.5
    # But we don't have cost_total directly. We have cost for 2000 candidates.
    # Actually, `cost` here is the cost of evaluating each of the 2000 candidates, not the total spent.
    # Wait, re-reading: "cost has shape (2000,)". So cost[i] is the cost of evaluating candidate i.
    # The total cost spent so far is not directly given. But we can estimate it.
    # Actually, the problem says "While total cost is below 30". The total cost is accumulated.
    # We don't have the total cost as an argument. We only have the per-candidate cost array.
    # 
    # Let me re-read: "mu, sigma and cost have shape (2000,). Return 2000 finite scores"
    # So cost[i] is the cost of evaluating the i-th candidate.
    # 
    # To infer progress, I can use the fact that the number of evaluations so far
    # determines the total cost. But I don't have that.
    # 
    # Alternative: use f_best as a proxy. As we find better points, f_best decreases.
    # But this is noisy.
    # 
    # Better approach: use a fixed schedule based on the expected budget.
    # Budget 30, initial 5 points (~7.5 cost), remaining ~22.5 cost.
    # Each eval ~1.5 cost, so ~15 more evals. Total ~20 evals.
    # 
    # For the acquisition function, I'll use a hybrid approach:
    # 1. Expected Improvement (EI) for exploitation
    # 2. Add a small exploration term that decays over time
    # 
    # But without knowing the exact eval count, I'll use a moderate fixed kappa.
    # 
    # Actually, let me try a different structural change: use Thompson Sampling-like
    # approach by sampling from the posterior, or use a different acquisition like
    # Probability of Improvement (PI) with cost normalization.
    # 
    # Let me try: score = (f_best - mu) / (cost * sigma_safe) + kappa * sigma_safe / cost_safe
    # This combines exploitation (low mu relative to f_best) with exploration (high sigma)
    # all normalized by cost.
    
    # PI-like term: how much improvement is expected
    improvement = np.maximum(f_best - mu, 0)
    
    # Normalize by cost and uncertainty
    # Use a combination of EI and exploration
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Cost-aware EI
    ei_cost = ei / cost_safe
    
    # Exploration bonus: reward high uncertainty, decayed by a factor
    # Use a moderate exploration weight
    exploration = 0.3 * sigma_safe / cost_safe
    
    score = ei_cost + exploration
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
