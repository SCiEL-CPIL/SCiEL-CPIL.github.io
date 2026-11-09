"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330 - 5  # Safety margin: kill program 5s before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Instead of using a deterministic bound like UCB or EI, we sample from the 
    posterior distribution of the objective function at each candidate point.
    For minimization, we want to find the point with the lowest potential value.
    
    The score for Thompson Sampling is typically the sampled value itself.
    However, we must incorporate the cost. The task specifies that evaluating x 
    costs $1 + \text{mean}(x_{scaled})$. The evaluator picks the candidate with 
    the "best" acquisition score. Usually, acquisition functions are designed so 
    that higher scores are better.
    
    If we use Thompson Sampling for minimization:
    1. Sample $\tilde{f}(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$.
    2. We want to select $x$ that minimizes $\tilde{f}(x) / \text{cost}(x)$? 
       Or minimize $\tilde{f}(x)$ subject to cost constraints?
       
    The prompt says: "The evaluator scores 2000 new uniform candidates with 
    acquisition and evaluates the best one." and "Return 2000 finite scores 
    (higher = evaluate next)".
    
    So we need a score $S(x)$ such that higher $S(x)$ means we should pick $x$.
    
    In standard BO with cost, one common approach is to maximize the Expected 
    Improvement per unit cost. For Thompson Sampling, we can approximate this 
    by sampling the function value and dividing by cost, but since we are 
    minimizing, we want the *lowest* value. 
    Therefore, the "best" candidate in terms of raw value is the one with the 
    smallest $\tilde{f}(x)$.
    
    To convert "minimize value" to "maximize score", we can use:
    $S(x) = - \frac{\tilde{f}(x)}{\text{cost}(x)}$
    
    This rewards points that have low sampled values and low costs.
    
    However, there is a nuance: Thompson Sampling is stochastic. We generate a 
    single sample for each of the 2000 candidates in this call. This introduces 
    randomness into the selection, which is the core benefit of TS for exploration.
    
    We must handle the deadline. If we are close to the deadline, we might 
    fall back to a faster, deterministic method like EI to ensure we return 
    valid scores in time, although numpy operations on 2000 points are very fast.
    """
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: Simple cost-aware EI (deterministic)
        # This is just a safety net to avoid timeout, not the primary strategy.
        from scipy.stats import norm
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    # Ensure sigma is positive for sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Generate random samples from the posterior distribution
    # Shape: (2000,)
    # rng = np.random.default_rng() # Using global np.random is fine for this scope
    samples = np.random.normal(mu, sigma_safe)
    
    # Calculate the score: we want to minimize the sampled value relative to cost.
    # Higher score = better.
    # Score = - (sampled_value / cost)
    scores = - (samples / cost_safe)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
