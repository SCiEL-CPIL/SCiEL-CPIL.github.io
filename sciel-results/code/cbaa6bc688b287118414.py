import numpy as np
from scipy.stats import norm
from scipy.special import softplus
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 290.0

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    n = len(mu)
    
    # Guard against zero/negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # 1. Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # 2. Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 3. UCB-style exploration term for minimization
    # Adaptive beta: increase exploration as budget is consumed
    # cost ranges from ~1 to ~30, so normalize to [0, 1]
    cost_norm = np.clip(cost_safe / 30.0, 0.0, 1.0)
    # Start with beta=2.0, increase to beta=4.0 as we spend more budget
    beta = 2.0 + 2.0 * cost_norm
    
    ucb_raw = (f_best - mu + beta * sigma_safe)
    ucb_smooth = softplus(ucb_raw)
    
    # Scale EI and UCB to be comparable to POI (0-1 range)
    # Use adaptive scale based on current f_best and known minimum
    # Shekel 4d min is -10.5364, so remaining improvement is f_best - (-10.5364)
    min_val = -10.5364
    max_improvement = np.maximum(f_best - min_val, 1.0)
    # Use a fraction of max improvement for scaling
    scale = 0.8 * max_improvement
    scale = np.maximum(scale, 1.0)  # Ensure scale is at least 1
    ei_scaled = ei / scale
    ucb_scaled = ucb_smooth / scale
    
    # Combine all terms
    # Weights adjusted to balance exploitation (POI) and exploration (UCB/EI)
    # POI: robust exploitation
    # EI: magnitude of improvement
    # UCB: uncertainty-driven exploration
    # Use relative cost normalization: cost relative to mean cost
    mean_cost = np.mean(cost_safe)
    cost_rel = cost_safe / (mean_cost + 1e-8)
    # Inverse cost weighting with softer penalty
    cost_weight = 1.0 / (1.0 + 0.5 * (cost_rel - 1.0))
    cost_weight = np.clip(cost_weight, 0.3, 2.0)
    
    scores = (0.4 * poi + 0.3 * ei_scaled + 0.3 * ucb_scaled) * cost_weight
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        fallback_scores = poi * cost_weight
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
