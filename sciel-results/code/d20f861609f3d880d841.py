"""
Deterministic time-gated hybrid LCB/EI acquisition with cost-awareness.

Strategy:
1. Compute LCB (Lower Confidence Bound) for minimization: LCB = mu - gamma * sigma.
2. Compute EI (Expected Improvement) for minimization.
3. Combine them with a weight that transitions from exploration (LCB) to exploitation (EI)
   based on the estimated progress of the budget.
   - Progress is estimated by looking at the cost of the current best candidate (f_best).
     Since cost = 1 + mean(x_scaled) and x_scaled in [0,1], cost is in [1, 2].
     However, f_best is the function value, not the cost. 
     Wait, the task says "cost" is passed as an array of costs for the 2000 candidates.
     We don't have the history of evaluated points' costs directly, but we can infer
     the "phase" of the search.
     
     Actually, a more robust indicator of "progress" is the magnitude of the improvement
     or the uncertainty. But the prompt suggests using the budget.
     Since we don't have the total spent budget directly, we can use a proxy:
     The minimum cost among the candidates is always ~1.0.
     The maximum cost is ~2.0.
     
     Alternative: Use the distribution of sigma. Early on, sigma is high everywhere.
     Later, sigma is low near evaluated points.
     
     Let's use a simpler, more direct approach based on the "cost-aware" nature.
     The cost of evaluating a point is 1 + mean(x_scaled).
     Cheaper points are those with lower mean(x_scaled).
     
     Let's use a fixed, robust schedule based on the number of calls? No, stateless.
     
     Let's use the ratio of the current best improvement potential to the initial scale.
     
     Actually, the most robust "time" proxy in a stateless function is the 
     "optimism" of the posterior. 
     
     Let's try a different structural change: 
     Instead of a complex hybrid, use a "Cost-Adjusted LCB" where the exploration
     coefficient gamma is inversely proportional to the cost.
     Score = (mu - gamma * sigma) / cost
     This naturally favors low-cost points for exploration and high-improvement points for exploitation.
     
     However, LCB can be negative (if mu is very low and sigma is high), which is good for minimization
     (we want to minimize the score? No, the task says "higher = evaluate next").
     So for minimization, we want to maximize the "potential for improvement".
     
     Standard EI is positive and higher is better.
     LCB (mu - gamma*sigma) is lower when it's better (more negative is better for the value, 
     but we want a score where higher is better).
     So we should use -LCB = -mu + gamma*sigma.
     
     Let's combine:
     Score = w * EI + (1-w) * (-LCB)
     where w increases as we get closer to the end of the budget.
     
     How to estimate "end of budget"?
     The total budget is 30.
     Initial points: 5 uniform. Cost ~ 1.5 each. Total ~ 7.5.
     Remaining budget ~ 22.5.
     Each new point costs ~ 1.5.
     So we have ~ 15 evaluations left.
     
     We can estimate the number of evaluations made by looking at the "spread" of the data?
     No, we only get mu, sigma, f_best, cost for the 2000 candidates.
     
     Let's use a simple heuristic: 
     If f_best is close to the theoretical minimum (0), we should exploit (high w).
     If f_best is far, we should explore (low w).
     But we don't know the theoretical minimum exactly (it's 0, but shifted).
     
     Let's use the magnitude of f_best. 
     If |f_best| is small, we are close to the minimum.
     
     Let's try: w = 1 / (1 + exp(-k * (threshold - |f_best|)))
     This is a sigmoid that shifts to exploitation as f_best gets close to 0.
     
     However, the function values of the 3-hump camel can be negative.
     The minimum is 0 (after shift? No, "min 0" in the task description means the global min is 0).
     So f_best <= 0 is possible if the function is shifted? 
     "min 0" likely means the global minimum value is 0.
     So f_best starts high (e.g., 10) and decreases towards 0.
     
     So we want w to increase as f_best decreases towards 0.
     w = 1 / (1 + exp(-k * (f_best - threshold)))
     If f_best is large (early), exp is large, w ~ 0 (Explore).
     If f_best is small (late), exp is small, w ~ 1 (Exploit).
     
     Let's use threshold = 1.0 (arbitrary, based on typical scale of 3-hump camel).
     k = 2.0.
     
     Also, we must divide by cost to be cost-aware.
     Score = (w * EI + (1-w) * (-LCB)) / cost
     
     Wait, -LCB = -mu + gamma*sigma.
     If mu is -10 and sigma is 1, -LCB = 10 + gamma.
     If mu is 0 and sigma is 1, -LCB = gamma.
     This seems reasonable.
     
     Let's refine the gamma.
     gamma = 1.0 (fixed) or adaptive?
     Let's use a fixed gamma = 1.5 for LCB.
     
     And for EI, use standard EI.
     
     Finally, ensure the score is finite and positive.
"""

import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def _ei(mu, sigma, f_best, eps=1e-9):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _lcb_score(mu, sigma, gamma=1.5):
    """
    Negative Lower Confidence Bound for minimization.
    We want to maximize this score.
    LCB = mu - gamma * sigma.
    To maximize potential improvement (minimize value), we look at -LCB = -mu + gamma * sigma.
    """
    return -mu + gamma * sigma

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a time-gated hybrid of LCB and EI.
    
    Strategy:
    1. Compute EI (exploitation) and -LCB (exploration).
    2. Determine the blend weight w based on f_best (proxy for progress).
       - Early (f_best high): w ~ 0, favor LCB (exploration).
       - Late (f_best low): w ~ 1, favor EI (exploitation).
    3. Combine scores and divide by cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute components
    ei_scores = _ei(mu, sigma_safe, f_best)
    lcb_scores = _lcb_score(mu, sigma_safe, gamma=1.5)
    
    # Determine blend weight w
    # f_best decreases as we find better points.
    # Assume a typical scale where f_best goes from ~10 to ~0.
    # Use a sigmoid to transition from exploration to exploitation.
    # w = 1 / (1 + exp(-k * (f_best - threshold)))
    # If f_best > threshold, w ~ 1 (Wait, if f_best is high, we want to explore, so w should be low for EI?)
    # Let's redefine:
    # Score = w * EI + (1-w) * LCB_score
    # We want w to be small when f_best is high (early).
    # w = 1 / (1 + exp(k * (f_best - threshold)))
    # If f_best is high (e.g., 10), exp is large, w ~ 0. (Good, explore)
    # If f_best is low (e.g., 0.1), exp is small, w ~ 1. (Good, exploit)
    
    threshold = 0.5  # Midpoint of expected f_best range
    k = 5.0          # Steepness of transition
    
    # Handle case where f_best is exactly 0 or very close
    f_best_val = float(f_best)
    w = 1.0 / (1.0 + np.exp(k * (f_best_val - threshold)))
    
    # Combine scores
    hybrid_score = w * ei_scores + (1.0 - w) * lcb_scores
    
    # Cost-awareness: divide by cost
    score = hybrid_score / cost_safe
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
