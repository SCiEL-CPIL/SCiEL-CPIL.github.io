import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Safety deadline guard
    deadline = start_time + 280.0
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Compute several candidate acquisition functions
    candidates = {}
    
    # 1. Pure EI / cost (the seed/baseline, score 1.0)
    candidates['ei_cost'] = ei / cost
    
    # 2. EI / cost^alpha for various alpha
    for alpha in [0.5, 0.75, 1.25, 1.5, 2.0]:
        candidates[f'ei_cost_{alpha}'] = ei / (cost ** alpha)
    
    # 3. Hybrid: (EI + kappa * sigma) / cost
    # This adds exploration via UCB-like term
    for kappa in [0.1, 0.5, 1.0, 2.0]:
        hybrid = (ei + kappa * sigma) / cost
        candidates[f'hybrid_{kappa}'] = hybrid
    
    # 4. Cost-aware UCB: (f_best - mu + beta * sigma) / cost
    for beta in [1.0, 2.0, 3.0]:
        ucb = (f_best - mu + beta * sigma) / cost
        candidates[f'ucb_{beta}'] = ucb
    
    # 5. Mixed EI and UCB
    for mix in [0.3, 0.5, 0.7]:
        mixed = mix * (ei / cost) + (1 - mix) * ((f_best - mu + 1.0 * sigma) / cost)
        candidates[f'mixed_{mix}'] = mixed
    
    # 6. Log-cost aware: EI / log(1 + cost)
    candidates['ei_logcost'] = ei / np.log(1 + cost)
    
    # 7. Square-root cost aware
    candidates['ei_sqrtcost'] = ei / np.sqrt(cost)
    
    # Select the best candidate based on a heuristic metric
    # We want the candidate that gives high scores to the most promising points
    # Metric: 95th percentile of scores (robust to outliers)
    # But also penalize candidates with too many very low scores
    
    best_score = -np.inf
    best_result = None
    
    for name, scores in candidates.items():
        # Check deadline
        if time.time() > deadline:
            break
        
        # Ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Metric: combination of 95th percentile and mean
        p95 = np.percentile(scores, 95)
        mean_val = np.mean(scores)
        metric = 0.7 * p95 + 0.3 * mean_val
        
        if metric > best_score:
            best_score = metric
            best_result = scores
    
    # If we ran out of time or no valid result, fall back to pure EI/cost
    if best_result is None:
        best_result = ei / cost
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Emergency fallback
        best_result = ei / cost
    
    # Ensure all values are finite
    best_result = np.nan_to_num(best_result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return best_result
