# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    
    Strategy: Quantile Expected Improvement (QOEI) with adaptive cost-weighting.
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return inverse cost (cheapest)
        return 1.0 / np.maximum(cost, 1e-10)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Quantile Expected Improvement (QOEI)
    # QOEI interpolates between EI (q=1) and a more aggressive quantile.
    # For minimization, we look at the distribution of f.
    # We want to maximize E[max(f_best - f, 0)] but with a twist that favors
    # regions where the probability of significant improvement is high.
    
    # Standard EI calculation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # QOEI term: 
    # A common variant is to use the quantile of the improvement.
    # Let's use a "soft" best-of-n approximation or a quantile of the predictive dist.
    # Here we use a weighted combination that emphasizes the tail of the improvement.
    # Specifically, we can approximate the expected max of n samples (which relates to quantiles).
    # For n=2, it's related to EI but sharper.
    # Let's define a QOEI score that is sensitive to the uncertainty in a different way.
    # q = 0.5 (median improvement) is less aggressive than EI.
    # Let's stick to a robust EI variant but adjust for cost more aggressively.
    
    # Alternative QOEI formulation:
    # Score = E[f_best - min(f_1, f_2, ... f_n)]
    # For n=1, it's EI.
    # Let's approximate n=2 QOEI which is more aggressive in exploration.
    # E[max(0, f_best - min(f1, f2))]
    # This is harder to compute analytically for arbitrary GP.
    # Instead, let's use a "Quantile" of the predictive distribution directly.
    # q-th quantile of f: mu + sigma * norm.ppf(q)
    # We want low quantiles (minimization).
    # Let's use the 10th percentile as a proxy for "aggressive" improvement potential.
    q_aggressive = 0.1
    q_ei = 0.5 # Median
    
    quantile_score = mu + sigma_safe * norm.ppf(q_aggressive)
    # Lower is better. So we negate it for "score".
    qoei_raw = -quantile_score
    
    # 2. Compute Cost Efficiency
    # Cost is in [1, 2].
    # We want to maximize value per cost.
    cost_eff = 1.0 / (cost + eps)
    
    # 3. Adaptive Weighting based on Budget
    # Total budget is 30. Start with 5 points (cost ~ 1.5 * 5 = 7.5? No, cost is 1+mean(x)).
    # Initial cost is roughly 1 + 0.5 = 1.5 per point. 5 points -> ~7.5 cost.
    # Remaining budget determines how much we can afford to explore (high cost) vs exploit (low cost).
    # If budget is high, we can afford higher cost points to find the minimum.
    # If budget is low, we must be very efficient (low cost).
    
    # Estimate remaining budget.
    # We don't know the exact current cost, but we can infer from the 'cost' array of candidates?
    # No, 'cost' is the cost of the *candidates*.
    # We need the *current* total cost. The function signature doesn't provide it directly.
    # However, the problem states "While total cost is below 30".
    # We can approximate the number of iterations or use a heuristic.
    # Let's assume we are in the middle of the process.
    # A safe heuristic: weight cost efficiency more heavily as we approach the end.
    # Without current total cost, we can use a fixed moderate weight, or try to estimate.
    # Actually, the prompt says "cost" argument is the cost of the candidates.
    # Let's look at the distribution of costs in the candidates to gauge the "price" of exploration.
    # But we need the *budget* state.
    
    # Heuristic: 
    # Early on: Explore (high weight on EI/QOEI, low weight on cost).
    # Late on: Exploit (high weight on cost efficiency, moderate EI).
    
    # Since we don't have the global step count, let's use a static but optimized blend.
    # The previous program used 0.7 EI / 0.3 UCB.
    # Let's try 0.6 QOEI / 0.4 Cost Efficiency.
    
    # Normalize QOEI and Cost Efficiency to [0, 1]
    qoei_min = np.min(qoei_raw)
    qoei_max = np.max(qoei_raw)
    qoei_range = qoei_max - qoei_min
    if qoei_range < eps:
        qoei_norm = np.ones_like(qoei_raw)
    else:
        qoei_norm = (qoei_raw - qoei_min) / qoei_range
        
    cost_min = np.min(cost_eff)
    cost_max = np.max(cost_eff)
    cost_range = cost_max - cost_min
    if cost_range < eps:
        cost_norm = np.ones_like(cost_eff)
    else:
        cost_norm = (cost_eff - cost_min) / cost_range

    # 4. Combine
    # Use a power mean or simple weighted sum.
    # Weighted sum is robust.
    w_qoei = 0.6
    w_cost = 0.4
    
    combined = w_qoei * qoei_norm + w_cost * cost_norm
    
    # 5. Final Adjustment
    # Ensure the best candidate has a score significantly higher than others to break ties
    # in favor of the one with the best combined metric.
    
    # Optional: Multiply by a small factor of EI to ensure we don't pick a point with 
    # zero improvement potential just because it's cheap.
    # If EI is 0, the point cannot improve f_best. We should penalize it.
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    # Gate the score by EI. If EI is very low, reduce the score.
    # This prevents picking cheap points that are unlikely to improve the best.
    gate = 0.5 + 0.5 * ei_norm
    
    final_scores = combined * gate
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores

# EVOLVE-BLOCK-END
