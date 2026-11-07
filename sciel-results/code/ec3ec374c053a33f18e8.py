"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Instead of deterministic Expected Improvement (EI), we use Thompson Sampling.
       This involves sampling a value from the posterior distribution N(mu, sigma^2) 
       for each candidate point.
       
    2. To make it cost-aware, we adjust the sampled value by the cost.
       The goal is to minimize the objective function, but evaluations are expensive.
       A common approach in cost-aware BO is to minimize the "regret per cost" or 
       simply use the sampled value but penalize high cost.
       
       However, a more robust "cost-aware" Thompson Sampling variant often used 
       implicitly in such tasks is to sample from the posterior and then select 
       the point that minimizes the sampled value, but we must return scores for 
       the 2000 candidates such that the *evaluator* picks the best one.
       
       Wait, the interface returns scores for 2000 candidates, and the external 
       evaluator picks the one with the highest score. 
       
       If we use Thompson Sampling, we typically sample once from the posterior 
       for each candidate. Let z_i ~ N(mu_i, sigma_i^2).
       Then we want to maximize the "utility" of picking candidate i.
       
       For minimization, lower f(x) is better.
       The sampled value z_i is an estimate of f(x_i).
       Lower z_i is better.
       
       But we have costs. 
       A standard cost-aware metric is Expected Improvement divided by cost, 
       but for Thompson Sampling, a natural analogue is to sample the value 
       and then divide by the cost? Or subtract cost?
       
       Let's look at the "inspiration 2" which scored 0.8348 (similar to current).
       It mentioned: "Adopt a Thompson Sampling strategy that samples a single 
       function from the GP posterior... and selects the candidate with the 
       lowest sampled value".
       
       If we just return -z_i / cost_i, we are maximizing "improvement per cost" 
       in a stochastic sense.
       
       Let's refine:
       1. Sample z_i = mu_i + sigma_i * u_i where u_i ~ N(0, 1).
       2. We want to find the point that is likely to have the lowest value 
          considering the cost.
          
       If we simply return -z_i / cost_i:
       - If z_i is very low (good), -z_i is high (good score).
       - If cost_i is high, the score is penalized.
       
       This effectively performs a stochastic search for the best value-per-cost.
       
       To ensure we don't just pick the absolute minimum z_i (which might be 
       noisy and high cost), dividing by cost helps.
       
       However, there is a risk: if z_i is positive (bad, since we minimize), 
       -z_i is negative. If z_i is negative (good), -z_i is positive.
       Rastrigin is non-negative. f_best starts high.
       
       Let's consider the scale. mu and sigma are in objective units.
       If mu is around 10 and sigma is 1, z is around 10. -z is -10.
       Scores should be higher = evaluate next.
       So we want candidates with LOW z to have HIGH scores.
       
       Score = -z_i / cost_i seems reasonable.
       
       But wait, if all z_i are positive (likely for Rastrigin unless we find the minimum), 
       all scores will be negative. That's fine, the relative ordering matters.
       
       One issue: Thompson Sampling is stochastic. In a loop, this means the 
       selection varies. This is good for exploration.
       
       To make it more robust, we can add a small deterministic component 
       or ensure the noise level is appropriate.
       
       Let's try:
       u = np.random.normal(0, 1, size=mu.shape)
       z = mu + sigma * u
       
       # We want to maximize score.
       # Lower z is better.
       # Higher cost is worse.
       
       # Score = -z / cost
       
       # To handle the case where z might be very large negative (unlikely in Rastrigin 
       # as min is 0, but possible if GP predicts negative due to noise/standardization),
       # we just rely on the ordering.
       
       # One refinement: 
       # Sometimes, pure TS can be too aggressive or too conservative.
       # But the prompt suggests a structural change to TS.
       
       # Let's add a deadline guard.
    """
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a neutral score, e.g., based on mu only, to finish quickly
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Generate random samples
    # Using a fixed seed is bad for TS (needs to be stochastic to explore).
    # The evaluator runs 5 instances. We want different exploration for each?
    # The function is called per iteration. The randomness should come from 
    # the global numpy state, which is seeded by the harness per instance?
    # The task says "five seeded instances". This implies the harness seeds 
    # numpy.random before running the loop for each instance.
    # So we should use np.random.normal directly.
    
    u = np.random.normal(0, 1, size=mu.shape)
    z = mu + sigma * u
    
    # Cost-aware score
    # We want to maximize this.
    # Lower z (better function value) -> Higher score.
    # Lower cost (cheaper) -> Higher score.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score = -z / cost
    # This penalizes high cost and rewards low sampled value.
    
    scores = -z / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
