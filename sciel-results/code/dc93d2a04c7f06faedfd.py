import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Treat the search as a stochastic bandit problem.
    2. Instead of deterministic Expected Improvement (EI) or UCB, we use
       Thompson Sampling. This involves drawing a sample from the posterior
       distribution of the function value at each candidate point.
    3. The score is defined as the "inverse regret" of this sample, adjusted
       for cost: Score = (f_best - sample) / (sample_cost + eps).
       
    Why this is different:
    - EI is deterministic and tends to be "greedy" in exploitation, often
      ignoring high-uncertainty regions unless they have very high mean improvement.
    - UCB adds a linear uncertainty bonus, which can be too aggressive or passive
      depending on the scaling.
    - Thompson Sampling (TS) naturally balances exploration and exploitation.
      By sampling, we occasionally pick high-uncertainty points (exploration)
      and usually pick low-value points (exploitation).
    - The cost adjustment is multiplicative (division by cost), which is
      structurally different from additive penalties (Lagrangian) or power-law
      adjustments. It directly models "value per unit cost".
    - No progress estimation is needed. The stochastic nature of TS adapts
      automatically: early on, sigma is high everywhere, so samples vary widely,
      leading to broad exploration. Later, sigma drops in explored areas, and
      samples concentrate on the best regions, leading to exploitation.
    """
    start_time = time.time()
    deadline = start_time + 250.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
        
    # Ensure sigma is positive for sampling
    sigma = np.maximum(sigma, 1e-6)
    
    # Check time before expensive operation
    if time.time() > deadline - 10.0:
        # Fallback to a simple fast score if we are running out of time
        # Just use EI-like score divided by cost
        z = (f_best - mu) / sigma
        z_clipped = np.clip(z, -10, 10)
        # Approximate EI without scipy for speed in emergency
        # Using a simple linear approximation for very small z? 
        # Actually, just return 0 or a simple heuristic to be safe.
        # But we must return finite scores.
        # Let's use a fast heuristic: (f_best - mu) / cost
        scores = (f_best - mu) / (cost + 1e-6)
        scores[~np.isfinite(scores)] = 0.0
        return scores

    # Thompson Sampling: Draw one sample from the posterior N(mu, sigma^2)
    # This is a vectorized operation.
    # Note: We draw a single sample for each of the 2000 points.
    # This effectively simulates one step of the bandit algorithm.
    # Since the outer loop picks the argmax, this is equivalent to
    # maximizing the sampled value.
    
    # Draw samples
    # Using standard normal random numbers
    z = np.random.standard_normal(n)
    samples = mu + sigma * z
    
    # The "regret" of a sample relative to the current best is f_best - sample.
    # If sample < f_best, regret is positive (improvement).
    # If sample > f_best, regret is negative (worse than current best).
    # We want to maximize improvement per cost.
    # Score = Improvement / Cost
    
    # To avoid division by zero and handle negative improvements gracefully:
    # We can clip the improvement to be non-negative? 
    # In standard TS for minimization, we usually pick the sample with the lowest value.
    # Here, the outer loop picks the MAX score.
    # So we want Score to be high when sample is low.
    # Score = 1 / (sample + offset) ? No, sample can be negative.
    # Score = f_best - sample ? This is improvement.
    # Cost-aware: Score = (f_best - sample) / cost.
    
    # If (f_best - sample) is negative, the score is negative.
    # This is fine, as the max will likely be positive if there's any improvement potential.
    # However, if all samples are worse than f_best, all scores are negative.
    # The loop will still pick the "least bad" one (max negative), which is correct 
    # behavior for a greedy step if no improvement is likely (it picks the most 
    # promising candidate relative to others, even if all are bad).
    
    # One issue: If sigma is very small, samples are close to mu.
    # If mu is close to f_best, samples are close to f_best.
    # The score (f_best - sample) / cost will be small.
    # This is consistent with EI behavior when no improvement is expected.
    
    # Let's compute the score
    improvement = f_best - samples
    scores = improvement / (cost + 1e-6)
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    # Optional: If we want to ensure that we don't pick a point that is clearly
    # worse than the best observed, we could clip the score at 0?
    # No, in TS, we should still pick the "best of the worst" if all are bad,
    # to maintain exploration or move to a new region.
    # However, if the GP is confident that no point is better than f_best,
    # the samples will all be > f_best, so scores < 0.
    # The max will be the least negative.
    # This is a valid strategy.
    
    # Check time
    if time.time() > deadline:
        return scores
        
    return scores
