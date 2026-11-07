"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_LIMIT = 325.0  # Safety margin before the 330 s wall-clock kill

def acquisition(mu, sigma, f_best, cost):
    """
    Budget-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample from the GP posterior: y ~ N(mu, sigma^2)
    2. Normalize the cost to reflect the remaining budget.
       The prompt states the loop runs while total cost < 30.
       We estimate the current total cost by summing the costs of the evaluated points.
       However, we are not given the history of evaluated points, only the current GP state.
       Wait, the prompt says: "Runs start with 5 uniform random points... While total cost is below 30...".
       The function `acquisition` is called in a loop. We don't have access to the total cost history directly 
       unless we infer it or assume a static normalization.
       
       Actually, looking at the inputs: mu, sigma, f_best, cost.
       `cost` is the cost of evaluating the 2000 candidate points, shape (2000,).
       It is NOT the total cost spent so far.
       
       To make it "budget-aware", we need to know how much budget is left.
       Since we don't have the total cost spent as an input, we have two options:
       A) Assume the cost normalization should be relative to the typical cost (mean cost).
       B) Infer the number of iterations. The loop runs until cost >= 30.
          Initial cost: 5 uniform points. Cost per point is 1 + mean(x_scaled).
          Mean x_scaled is 0.5, so mean cost is 1.5.
          5 points * 1.5 = 7.5 initial cost.
          Remaining budget ~ 22.5.
          Average number of additional evaluations ~ 22.5 / 1.5 = 15.
          Total evaluations ~ 20.
          
       If we use a static "remaining budget" estimate, it might be inaccurate later.
       However, Thompson Sampling with a cost penalty that is inversely proportional to the 
       "value" of the evaluation (which is higher when budget is scarce) is a good heuristic.
       
       A simpler, robust approach for cost-aware TS:
       Score = -sampled_value / cost
       
       This is essentially "Expected Improvement per unit cost" but with the randomness of TS.
       The randomness of TS helps escape local optima more effectively than EI in some cases,
       especially with the cost penalty which might create a "ratchet" effect in EI/cost.
       
       Let's refine the cost penalty.
       If we just use -y / cost, we are maximizing the expected reduction in objective per cost.
       
       Let's try to be slightly more sophisticated.
       Standard TS for minimization: sample y, pick min y.
       Cost-aware TS: pick min (y + lambda * cost) or min (y / cost)?
       Minimizing y/cost is equivalent to maximizing -y/cost.
       
       Let's use: score = -sampled_value / cost.
       This is a stochastic version of EI/cost.
       
       To add a "budget-aware" twist without knowing the exact remaining budget:
       We can assume that as the search progresses, the "price" of a unit of objective improvement increases.
       However, without iteration count, we can't easily model this.
       
       Alternative: Use the variance of the GP to inform the exploration.
       Thompson Sampling naturally balances this.
       
       Let's stick to the core difference: **Thompson Sampling** instead of **EI**.
       EI is deterministic (given GP). TS is stochastic.
       The stochasticity can help break symmetry or get stuck in regions where EI is flat.
       
       Implementation:
       1. Generate samples from Normal(mu, sigma).
       2. Calculate score = -sample / cost.
       3. Return scores.
       
       Wait, is -y/cost the best metric?
       If y is very negative (good), -y is positive.
       If cost is high, score is lower.
       This encourages finding good values cheaply.
       
       Let's add a small epsilon to cost to avoid division by zero (though cost >= 1).
       
       One potential issue: If the GP is poorly calibrated, TS might sample extreme values.
       We should clip the samples or scores to ensure finiteness and stability.
       
       Let's also consider the "standardized targets" note.
       "mu and sigma ... are in objective units."
       So we don't need to un-standardize.
       
       Let's implement Thompson Sampling with cost normalization.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes and safety
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: if we are close to the limit, fall back to a fast deterministic score
    # Using the mean (mu) instead of sampling is faster and deterministic.
    if time.time() - _START > _LIMIT:
        # Deterministic fallback: -mu / cost
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, 0.0)

    # Thompson Sampling: Draw a sample from the posterior distribution
    # y ~ N(mu, sigma^2)
    # We use a single sample for all 2000 points. This is standard for TS in this context.
    # Using the same random seed for all calls? No, numpy random state is global.
    # We just draw random numbers.
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Cost-aware score: Minimize y / cost -> Maximize -y / cost
    # This is a stochastic version of the cost-effective objective.
    scores = -samples / cost
    
    # Clip scores to prevent extreme values from dominating due to random noise
    # The scale of Griewank is small (min 0, max ~ 1.5 in this range).
    # Scores should be in a reasonable range.
    # Let's clip to [-100, 100] to be safe, though they should be much smaller.
    scores = np.clip(scores, -100.0, 100.0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
