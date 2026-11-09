"""
Thompson Sampling acquisition for cost-aware Bayesian optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Strategy:
    1. Interpret mu and sigma as the mean and standard deviation of the GP posterior.
    2. Sample a single realization of the function values from N(mu, sigma^2) for all 2000 candidates.
    3. Compute the cost-normalized score as -sample_value / cost.
       (We minimize the objective, so lower sample values are better.
        We divide by cost to account for the cost-aware constraint.
        We negate so that higher scores are better for the 'max' selection in the evaluator loop).
    4. Return the scores.
    
    Note: 
    - f_best is not directly used in the score calculation for TS, but could be used for 
      early termination or normalization if needed. Here we rely on the raw sampled values 
      relative to their cost.
    - The 'cost' is 1 + mean(x). Since x is in [0,1]^3, cost is in [1, 2].
    - We want to minimize f(x). So a low sampled f(x) is good.
    - We want to minimize cost. So a low cost is good.
    - Score = - (sampled_f / cost). Higher score = better.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against invalid values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Check time budget
    if time.time() > deadline:
        # Fallback to simple EI if we are running out of time, though TS is very fast
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / cost_safe

    # Thompson Sampling: Draw one sample from the posterior for each candidate
    # Using a fixed seed is bad for TS as it removes the exploration randomness across calls?
    # No, the evaluator calls this function once per iteration.
    # The randomness in TS comes from the sampling. 
    # We should use the global random state or a local one? 
    # Using np.random.normal is standard.
    
    # Sample from N(mu, sigma)
    # To ensure reproducibility in a single call but randomness across calls, 
    # we just call np.random.normal.
    
    # However, if we want to be extra safe against "lucky" or "unlucky" draws in a single batch,
    # standard TS draws one sample. 
    # Let's draw one sample for all 2000 candidates.
    
    samples = mu + sigma_safe * np.random.normal(size=mu.shape)
    
    # We want to minimize the objective. 
    # So we want to select candidates with LOW sample values.
    # The evaluator picks the candidate with the HIGHEST score.
    # So score should be inversely related to the sample value.
    # score = -sample / cost
    
    score = -samples / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
