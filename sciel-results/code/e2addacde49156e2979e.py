"""Rewritten program with a dynamic, cost-aware GP-LCB acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Track time to ensure we stay well within limits, though vectorized ops are fast
    start_time = time.time()
    
    # 1. Safety handling for sigma
    # Avoid division by zero and ensure numerical stability
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # 2. Dynamic Exploration Coefficient (Beta)
    # The task states: "While total cost is below 30...". 
    # We can estimate the current progress based on the 'cost' array provided.
    # The cost array contains the cost of the 2000 candidates being evaluated.
    # However, we don't have the *total* cost accumulated so far directly as a scalar.
    # But we can infer the "urgency" or stage of optimization from the provided data.
    # A robust heuristic: If we assume the loop runs until cost ~30, 
    # we can use a fixed but tuned beta, or try to estimate remaining budget.
    # Since we don't have the global state (total cost so far), we must rely on the 
    # properties of the inputs. 
    # 
    # Alternative Strategy: 
    # Instead of trying to guess the global time step (which we don't have), 
    # we can use a "Cost-Aware LCB" that naturally balances exploration.
    # 
    # Let's use a Beta that is slightly higher than the standard 1.5 to encourage 
    # more exploration in the early/mid game, but we will penalize cost more heavily.
    # 
    # Actually, a better "different strategy" is to use the **Knowledge Gradient (KG)** 
    # or a **Thompson Sampling** approximation? 
    # Thompson Sampling is hard to vectorize efficiently for 2000 points without sampling 
    # the GP posterior many times. 
    # 
    # Let's stick to a refined LCB but change the COST MODEL.
    # Current: score = (EI + 0.2*LCB) / cost
    # New: score = LCB / (cost ** 2)
    # Why? In 4D, the cost range is [1, 2]. cost^2 ranges [1, 4]. 
    # This significantly penalizes points that are "expensive" (high x values).
    # Since the minimizer is in the inner 80% (roughly center), and cost increases 
    # with x, penalizing high x helps steer towards the center/low-cost regions 
    # where the minimum is likely to be, or at least prevents wasting budget 
    # on expensive regions that might have high uncertainty but high cost.
    
    # 3. Compute GP-LCB for minimization
    # LCB = mu - beta * sigma
    # We want to maximize the score, so we maximize (-mu + beta * sigma)
    
    # Let's try a beta that adapts slightly based on the spread of sigma or f_best?
    # No, keep it simple but effective. 
    # The previous blend (0.8 EI + 0.2 LCB) was good. 
    # Let's try a pure LCB with a higher beta, but with a stronger cost penalty.
    
    beta = 2.0  # Increased exploration compared to 1.5
    
    lcb_value = mu - beta * sigma_safe
    # We want to find low lcb_value. 
    # Score should be high for low lcb_value.
    # score_raw = -lcb_value = -mu + beta * sigma_safe
    
    score_raw = -mu + beta * sigma_safe
    
    # 4. Cost-Aware Scaling
    # Cost is 1 + mean(x_scaled). 
    # We want to discourage high cost.
    # Using cost^2 amplifies the difference between cost=1 and cost=2 (factor of 4 vs factor of 2).
    # This encourages the optimizer to stay in "cheap" regions.
    # Given the minimizer is in the inner 80% of [-4, 5], which is [-2.8, 3.8].
    # Scaled to [0,1]: 
    # -4 -> 0, 5 -> 1. Range 9.
    # -2.8 -> (1.2)/9 = 0.133
    # 3.8 -> (7.8)/9 = 0.866
    # The "center" of the box is x=0.5 (orig x=0.5).
    # The cost is linear in x.
    # Points near x=0 (scaled ~0.44) have cost ~1.44.
    # Points near x=5 (scaled 1.0) have cost 2.0.
    # Points near x=-4 (scaled 0.0) have cost 1.0.
    # The minimum is in the inner 80%, so it's not at the extreme low cost edge (-4) 
    # nor the extreme high cost edge (5). It's somewhere in the middle.
    # However, the function is powell, which is generally smooth.
    
    # Let's use a softer penalty: cost ** 1.5
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    scores = score_raw / (cost_safe ** 1.5)
    
    # 5. Fallback / Safety
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        # Fallback to simple EI/cost if it's taking too long (unlikely)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / cost_safe

    return scores
# EVOLVE-BLOCK-END
