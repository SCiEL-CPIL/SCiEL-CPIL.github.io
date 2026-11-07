"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Uses Expected Improvement (EI) with a linear cost penalty that increases
    as the best observed value improves. This shifts the search from exploration
    (tolerating higher cost) to exploitation (penalizing cost heavily) as the
    budget depletes and the solution converges.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: Linear penalty in objective space
    # We want to subtract a penalty proportional to cost.
    # The penalty strength (lambda) should increase as we get closer to the optimum.
    # 
    # Heuristic for lambda:
    # - When f_best is high (early stage), EI is large. We can afford a small lambda.
    # - When f_best is low (late stage), EI is small. We need a large lambda to 
    #   prioritize cheap points to conserve budget.
    #
    # Let's scale lambda based on the improvement relative to a baseline.
    # The Powell function minimum is 0. A typical initial random value might be ~10-50.
    # We want lambda to be small when f_best is large, and large when f_best is small.
    
    # A simple adaptive strategy:
    # lambda = base_lambda * (1 + scale * (1 / (f_best + epsilon)))
    # However, f_best can be 0. Let's use a saturation function.
    # 
    # Alternative: Use the ratio of current EI magnitude to cost magnitude?
    # No, the knowledge suggests a simple linear penalty with progress-based scaling.
    
    # Let's estimate the "progress". 
    # If f_best is close to 0, we are in exploitation mode.
    # If f_best is far from 0, we are in exploration mode.
    
    # Scale factor: 
    # At f_best = 10, lambda should be ~0.1 (mild penalty)
    # At f_best = 1, lambda should be ~1.0 (strong penalty)
    # At f_best = 0.1, lambda should be ~10.0 (very strong penalty)
    
    # lambda = c / (f_best + c) ? No, that's bounded.
    # lambda = c * exp(-f_best / d) ?
    
    # Let's try: lambda = base / (f_best + small)
    # If f_best = 50, lambda = 1/50 = 0.02. Cost penalty for cost=2 is 0.04. EI might be 5. OK.
    # If f_best = 1, lambda = 1/1 = 1.0. Cost penalty for cost=2 is 2.0. EI might be 0.5. 
    #    Score = 0.5 - 2.0 = -1.5. This strongly discourages expensive points.
    # If f_best = 0.01, lambda = 1/0.01 = 100. Cost penalty = 200. EI ~ 0. 
    #    This forces selection of the absolute cheapest points among those with similar EI.
    
    # We need to tune 'base'. 
    # Let's look at the cost range: [1, 2].
    # Let's look at typical EI values.
    
    base_lambda = 0.5
    epsilon = 1e-6
    
    # Adaptive lambda
    # We want lambda to grow as f_best decreases.
    # lambda = base_lambda / (f_best + epsilon)
    # To prevent lambda from exploding if f_best is extremely small (numerical issues), 
    # we can cap it or use a softer function.
    
    # Let's use: lambda = base_lambda * (1 + 10 * exp(-f_best))
    # If f_best = 10, exp(-10) ~ 0, lambda ~ 0.5.
    # If f_best = 1, exp(-1) ~ 0.36, lambda ~ 0.5 * (1 + 3.6) ~ 2.3.
    # If f_best = 0, exp(0) = 1, lambda ~ 0.5 * 11 = 5.5.
    
    # This seems reasonable. The penalty at f_best=0 is 5.5 * cost.
    # Max cost is 2, so max penalty is 11.
    # If EI is ~0.1, we prefer cost=1 (penalty 5.5) over cost=2 (penalty 11) by 5.5 units.
    # This is a strong preference.
    
    # Let's refine the base_lambda. 
    # If we start with f_best ~ 20, lambda = 0.5. 
    # Penalty for cost=2 is 1.0. 
    # If EI is 5.0, score is 4.0. 
    # If a cheaper point has EI 4.0, score is 4.0 - 0.5*1 = 3.5.
    # So we prefer the expensive point with higher EI. Good for exploration.
    
    # As f_best drops to 1.0, lambda = 2.3.
    # Penalty for cost=2 is 4.6.
    # If EI is 0.5, score = 0.5 - 4.6 = -4.1.
    # If a cheaper point has EI 0.4, cost=1, score = 0.4 - 2.3 = -1.9.
    # We prefer the cheaper point. Good for exploitation.
    
    lambda_val = base_lambda * (1.0 + 10.0 * np.exp(-f_best))
    
    # Calculate score
    score = ei - lambda_val * cost
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # If all are -inf (should not happen), return zeros
    if not np.any(np.isfinite(score)):
        score = np.zeros_like(score)
        
    return score
# EVOLVE-BLOCK-END
