import numpy as np
import time
from scipy.stats import norm

# Global state for deadline guard and progress estimation
_start_time = None
_n_evals = 5  # Start with 5 initial points
_avg_cost_estimate = 1.5  # Rough average cost per eval (1 + 0.5)

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _n_evals
    
    # Initialize time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are close to the 330s limit, return safe exploitation scores
    # to avoid any potential overhead or instability near the cutoff.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Simple exploitation: favor low mu, penalized by cost
        # f_best - mu is the improvement.
        scores = (f_best - mu) / np.maximum(cost, 1e-9)
        # Clip to ensure finite
        scores = np.clip(scores, -1e6, 1e6)
        return scores

    # Update internal state
    _n_evals += 1
    
    # Estimate total cost so far
    total_cost = _n_evals * _avg_cost_estimate
    
    # Safety bounds for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (Minimization)
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    # Clip Z to avoid numerical issues in norm.cdf/pdf
    Z = np.clip(Z, -10.0, 10.0)
    EI = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    # EI can be slightly negative due to numerical error; clip to 0
    EI = np.maximum(EI, 0.0)
    
    # 2. UCB (Minimization)
    # Progress based on estimated budget usage
    progress = min(1.0, total_cost / 30.0)
    # Beta decays from 3.0 to 1.0
    beta = 3.0 - 2.0 * progress
    UCB = f_best - mu + beta * sigma_safe
    
    # 3. Mix EI and UCB
    # Early: more UCB (exploration). Late: more EI (exploitation).
    # w_ucb decays from 1.0 to 0.2
    # w_ei increases from 0.2 to 1.0
    w_ucb = 1.0 - 0.8 * progress
    w_ei = 0.2 + 0.8 * progress
    base_score = w_ucb * UCB + w_ei * EI
    
    # 4. Cost-Risk Adjustment
    # Risk of budget exhaustion increases as (total_cost + candidate_cost) approaches 30
    # Use exponential decay: exp(-lambda * projected_cost / 30)
    # This structurally down-weights candidates that threaten to exceed the budget limit
    lambda_risk = 3.0
    projected_cost = total_cost + cost_safe
    risk_factor = np.exp(-lambda_risk * projected_cost / 30.0)
    
    # Final Score
    scores = base_score * risk_factor
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
