import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    1. Use LCB = mu - beta * sigma. This favors regions with low mean 
       (exploit) or high uncertainty (explore). It is structurally different 
       from EI which integrates the tail of the distribution.
    2. Cost awareness: The cost is 1 + mean(x_scaled). In 8D, x_scaled in [0,1].
       Cost ranges [1, 2]. 
       We penalize high cost. A simple division is used in the seed. 
       Here we use a power penalty: score = LCB / (cost ** 1.5). 
       This makes the cost penalty more aggressive for high-cost points, 
       potentially steering search towards the cheaper half of the space 
       where the minimizer is likely to be found faster given the 
       cosine8 landscape structure (though the minimizer is moved, 
       the cost structure is fixed relative to the coordinate frame).
       
       Actually, since the minimizer is moved to a uniform point in the 
       inner 80%, the "cheap" region (low x) might not contain the minimum. 
       However, cost-aware BO usually benefits from reducing the cost of 
       evaluations to fit more evaluations in the budget. 
       
       Let's stick to a robust LCB. Beta is typically tuned. 
       A common heuristic for LCB in BO is beta = 2 * sqrt(log(1 + n)) or similar. 
       Here n is not passed, but we have 2000 candidates. 
       We can estimate an adaptive beta or use a fixed reasonable value. 
       Given the standardized targets, sigma is in objective units. 
       
       Let's use a fixed beta that balances exploration well for 8D. 
       beta = 2.0 is a common starting point. 
       
       Wait, the current seed uses EI. EI is generally more robust than LCB 
       in high dimensions if not tuned well. LCB can be too greedy or too 
       exploratory. 
       
       Alternative structural change: **Knowledge Gradient (KG)** approximation? 
       Too slow. 
       
       **Expected Hypervolume Improvement (EHVI)**? Not applicable for single objective.
       
       Let's try **UCB (Upper Confidence Bound) for minimization** which is 
       LCB. 
       
       To make it materially different and potentially better: 
       The seed uses `ei / cost`. 
       Let's try **EI with a cost-aware beta** or **LCB with a different cost penalty**.
       
       Actually, a very effective structural change for cost-aware BO is to 
       use the **cost-adjusted acquisition** where we consider the *expected 
       improvement per unit cost* more aggressively, or use a **Bayesian 
       Optimization with a cost-aware prior**. 
       
       However, we don't have access to the prior construction, only mu/sigma.
       
       Let's look at the cost structure: `1 + mean(x_scaled)`. 
       The minimum cost is 1 (at x=0). Max is 2 (at x=1). 
       The minimizer is in the inner 80% of the range. 
       
       If we use LCB:
       lcb = mu - beta * sigma
       score = lcb / (cost ** 2)  # Stronger penalty for cost
       
       Let's try this. If LCB is too aggressive, it might jump around. 
       EI is smoother. 
       
       Another structural change: **Probability of Improvement (PI)**? 
       PI = Phi(z). It's binary-like, less informative. 
       
       Let's stick to LCB but with a **dynamic beta** based on the spread of 
       mu/sigma or just a well-tuned constant. 
       
       Actually, the most distinct and potentially powerful change is to use 
       **Expected Improvement (EI) but with a *cost-dependent* improvement 
       threshold**. 
       
       No, the simplest structural change that often outperforms EI in 
       multi-dimensional spaces with cost is **LCB**. 
       
       Let's implement LCB with beta = 2.5 and cost penalty `cost ** 1.5`.
    """
    # Check time
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Return a safe, finite fallback (e.g., uniform random or zero)
        # Returning zeros is safe but might not be optimal. 
        # Returning the first candidate's score replicated is also safe.
        # Let's return a small constant to avoid biasing towards index 0.
        return np.full_like(mu, 0.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB for minimization: we want low values.
    # High acquisition score = good.
    # So we want to maximize (mu - beta * sigma).
    # This is equivalent to minimizing (mu + beta * sigma) if we flip signs, 
    # but here we return scores where higher is better.
    # So score = -(mu + beta * sigma) ? No.
    # If mu is low, we want high score.
    # If sigma is high, we want high score.
    # LCB_value = mu - beta * sigma.
    # We want to maximize LCB_value.
    beta = 2.0
    
    lcb = mu - beta * sigma_safe
    
    # Cost penalty
    # Cost is in [1, 2].
    # We divide by cost to penalize.
    # Using a power > 1 increases the relative penalty of high-cost points.
    cost_penalty = cost ** 1.5
    
    score = lcb / cost_penalty
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
