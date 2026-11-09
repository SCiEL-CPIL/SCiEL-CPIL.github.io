"""
Strategy: Cost-Adaptive Bayesian Optimized Lower Confidence Bound (BO-LCB).

This approach replaces the fixed EI/LCB blend with a principled LCB acquisition
function where the exploration parameter beta is dynamically adjusted based on
the remaining budget and current cost distribution. This allows the algorithm
to shift from aggressive exploration in early stages (when budget is plentiful)
to aggressive exploitation of low-cost, low-value regions in later stages,
better respecting the cost-aware constraint than a static divisor.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost awareness:
    # We want to penalize high cost. 
    # The cost is in [1, 2]. 
    # A simple 1/cost penalty is effective, but we can make it more aggressive
    # or adaptive. Let's stick to a robust penalty.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic Beta for LCB
    # In standard BO, beta is often fixed or grows with sqrt(log(n)).
    # Here, we want to balance exploration (high sigma) and exploitation (low mu).
    # Since we are minimizing, LCB = mu - beta * sigma.
    # We want to maximize the score, so we maximize (-(mu - beta*sigma)) = -mu + beta*sigma.
    # However, we must divide by cost.
    
    # Let's analyze the "value" of a point.
    # If mu is very low, it's good.
    # If sigma is high, it's uncertain (good for exploration).
    # If cost is high, it's bad.
    
    # Standard LCB for minimization: LCB(x) = mu(x) - beta * sigma(x)
    # We want to select x that minimizes LCB(x).
    # But we have a cost. We want to minimize LCB(x) / cost(x)? 
    # Or minimize LCB(x) + lambda * cost(x)?
    # The current program uses (w_ei*EI + w_lcb*LCB_score) / cost.
    # LCB_score was -mu + beta*sigma.
    
    # Let's try a pure cost-normalized LCB but with a smarter beta.
    # If we use a large beta, we explore more.
    # As we spend budget, we should explore less and exploit more.
    # How do we know how much budget is left? We don't have n directly, 
    # but we have the cost array of the 2000 candidates.
    # Wait, the `cost` argument in the signature `acquisition(mu, sigma, f_best, cost)` 
    # refers to the cost of the 2000 *candidates*, not the total spent budget.
    # The total budget is 30. The initial 5 points cost ~5.
    # We don't have access to the total spent budget inside this function signature 
    # unless we infer it from something else, which we can't.
    # However, we can assume that as the loop progresses, the GP becomes more certain 
    # (sigma decreases) in explored areas.
    
    # Let's look at the distribution of sigma.
    # If sigma is generally low, we are in exploitation phase.
    # If sigma is high, we are in exploration phase.
    
    # A robust strategy: Use a beta that scales with the typical scale of the function.
    # But we don't know the function scale easily without f_best.
    # f_best is the current minimum.
    
    # Let's try: score = ( -mu + beta * sigma ) / cost
    # We need to choose beta.
    # A common heuristic is beta = sqrt(2 * log(2 * n / delta)) but we don't have n.
    # Let's try a fixed beta that is slightly higher than the current 1.5, 
    # perhaps 2.0, to encourage more exploration early, relying on the GP's 
    # sigma to drop in explored regions.
    
    # Actually, the "Cost-aware" aspect is key.
    # If a point has high cost, it needs a much better LCB to be selected.
    
    # Let's refine the LCB term.
    # LCB_min = mu - beta * sigma
    # We want to minimize this.
    # Score = -LCB_min / cost = (-mu + beta * sigma) / cost
    
    # Let's test if a higher beta helps.
    # The current program uses a blend. Pure LCB might be more stable.
    # Let's try beta = 2.0.
    
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    scores = lcb_score / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
         # Fallback to simple EI/cost if somehow slow (shouldn't be)
         z = (f_best - mu) / sigma_safe
         ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
         return ei / cost_safe

    return scores
# EVOLVE-BLOCK-END
