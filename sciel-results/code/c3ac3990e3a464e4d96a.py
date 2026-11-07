import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability-Weighted Improvement per Cost.
    
    Structural change:
    - Uses P(mu < f_best) * (f_best - mu) / cost instead of LogPI.
    - This is a stochastic knapsack formulation: directly maximizes expected
      regret reduction per unit cost.
    - The probability term down-weights high-cost candidates with marginal
      improvement potential, while the improvement magnitude rewards
      high-confidence gains.
    - No arbitrary exploration weight needed; the probability term naturally
      balances exploration vs exploitation.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:
        # Fallback: simple heuristic if time is running out
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1.0)
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)

    # Calculate z-score: (f_best - mu) / sigma
    # This is the standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -20.0, 50.0)
    
    # Compute probability of improvement: P(y < f_best) = Phi(z)
    # Use norm.cdf for accuracy
    prob_improve = norm.cdf(z_clipped)
    
    # Ensure probability is in [0, 1] and finite
    prob_improve = np.clip(prob_improve, 0.0, 1.0)
    prob_improve = np.where(np.isfinite(prob_improve), prob_improve, 0.0)
    
    # Expected improvement magnitude (not full EI, just the mean of improvement)
    # E[max(f_best - y, 0)] where y ~ N(mu, sigma^2)
    # This equals sigma * (phi(z) + z * Phi(z))
    # But for the stochastic knapsack formulation, we use:
    # P(improve) * E[improvement | improve] = P(improve) * (f_best - mu + sigma * phi(z)/Phi(z))
    # Simplified: just use P(improve) * (f_best - mu) as a proxy
    # This is computationally cheaper and captures the key trade-off
    
    # The improvement magnitude (how much better than f_best)
    improvement = f_best - mu
    
    # Only consider positive improvements (points that could be better)
    improvement = np.maximum(improvement, 0.0)
    improvement = np.where(np.isfinite(improvement), improvement, 0.0)
    
    # Probability-weighted improvement per cost
    # This directly optimizes expected regret reduction per unit cost
    score = prob_improve * improvement / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
