import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Infer number of evaluations from total cost
    # Each evaluation costs 1 + mean(x) where x in [0,1]^4, so avg cost ~ 1.5
    # Total cost budget is 30, starting with 5 points
    # n_evals ≈ total_cost / avg_cost_per_eval
    # But we don't have total_cost directly; we can estimate from the fact that
    # the loop runs while total_cost < 30
    # Use a simpler heuristic: early iterations have lower cost sum
    # Estimate iteration number from f_best magnitude or just use a fixed schedule
    # Better: use the spread of mu/sigma to infer maturity
    # For now, use a simple decay based on how close we are to budget
    # Since cost is per-candidate, not cumulative, we need another signal
    # Use the number of distinct mu values as a proxy for iterations? No, mu is 2000 candidates
    
    # Alternative: use the variance of mu as a maturity indicator
    # Early on, GP is uncertain (high sigma relative to mu spread)
    # Late on, GP is confident (low sigma)
    # Compute a maturity metric
    mu_std = np.std(mu)
    if mu_std < 1e-9:
        maturity = 0.0
    else:
        # Normalize: when sigma is small relative to mu spread, we're mature
        sigma_mean = np.mean(sigma)
        maturity = 1.0 / (1.0 + sigma_mean / (mu_std + 1e-9))
    
    # Adaptive exploration bonus: higher early, lower late
    # bonus = base * (1 - maturity)^power
    # This encourages exploration when GP is uncertain
    base_bonus = 0.5
    power = 1.0
    exploration_bonus = base_bonus * (1.0 - maturity) ** power
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to maximize (EI + exploration_bonus * sigma) / cost
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus weighted by uncertainty
    scores = (ei + exploration_bonus * sigma) / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard: return immediately if we've used too much time
    if time.time() - start_time > 295.0:
        return scores
    
    return scores
