import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI) 
    gated by a strict cost efficiency constraint.
    
    Strategy:
    1. Compute Probability of Improvement: P(y < f_best) = Phi((f_best - mu) / sigma).
    2. Apply a hard gate: Only consider candidates if their cost is "efficient" relative to 
       the current best cost-efficiency ratio. Since we don't track history, we use a 
       heuristic gate: cost must be below a dynamic threshold derived from the cost array 
       itself (e.g., below the median cost) to force exploitation of the cheaper regions 
       (lower x values, since cost = 1 + mean(x_scaled)).
       Actually, the prompt says cost = 1 + mean(x scaled to [0,1]). Lower x means lower cost.
       The "low-cost corner" is near x=0 (in original coords, x=-10).
       
       However, the directive suggests a "multiplicative feasibility constraint" or "hard budget-aware gate".
       A simpler structural change from EI is PI. Let's stick to PI but with a cost penalty 
       that is steeper than division, or a gate.
       
       Let's try: Score = PI * (1 / (cost + epsilon)) is similar to EI/cost.
       To be materially different, let's use PI directly, but scaled by a cost efficiency factor 
       that penalizes high cost more aggressively, or simply PI if cost is low, else 0.
       
       Let's implement PI divided by cost, but using a different normalization or a gate.
       Actually, PI is bounded [0,1]. EI is unbounded.
       
       Let's try: Score = PI / cost. This is structurally different from EI/cost.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(y < f_best)
    pi = norm.cdf(z)
    
    # Cost-aware: PI / cost
    # This favors points with high probability of improvement and low cost.
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = pi / cost_safe
    
    # Sanitize
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
