import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling with Cost-Aware Normalization.
    
    Strategy:
    1. Draw a single sample from the GP posterior: y = mu - sigma * z.
       This represents a possible realization of the function values.
    2. Normalize the sample values to a common scale (e.g., standard score relative to the current best sample).
    3. Apply a cost-aware penalty to encourage exploring cheaper regions when improvement potential is similar.
    4. Return the score (higher is better).
    
    Why this may work:
    - Thompson Sampling provides a principled way to balance exploration and exploitation.
    - By sampling, we naturally explore regions with high uncertainty (high sigma).
    - The cost-aware normalization helps break ties in favor of cheaper evaluations,
      which is crucial given the cost structure (1 + mean(x_scaled)).
    - This is a structural change from deterministic acquisition functions (EI, PI, UCB)
      to a stochastic sampling-based approach.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw a single sample from the posterior
    # y ~ N(mu, sigma^2)
    z = np.random.standard_normal(len(mu))
    y = mu - sigma * z  # Note: minimizing, so we sample from N(mu, sigma^2)
    
    # We want to select the candidate that minimizes y / cost (or similar)
    # But the task says: "selecting the candidate that minimizes this sample divided by its cost"
    # Wait, the directive says: "selecting the candidate that minimizes this sample divided by its cost"
    # But the return interface requires returning 2000 scores, higher = evaluate next.
    # So we need to return a score such that the candidate with the lowest y/cost gets the highest score.
    # Score = - (y / cost) or Score = -y * (1/cost)
    
    # Let's compute the ratio: r = y / cost
    # Since we want to minimize r, we return -r as the score.
    # However, y can be very large or small, and cost is in [1, 2].
    # To make the score more robust, let's normalize y first.
    
    # Alternative: The directive says "minimizes this sample divided by its cost".
    # So the "value" to minimize is v = y / cost.
    # The acquisition score should be high for low v.
    # score = -v = -y / cost
    
    # But we also need to consider that y is in objective units, which can be large.
    # Let's normalize y by the current f_best to make it scale-invariant.
    # However, f_best is the best observed value, not the best sampled value.
    
    # Let's try: score = -(y - f_best) / cost
    # This penalizes samples that are worse than f_best, and rewards those that are better.
    # And it divides by cost, so cheaper points get a higher score for the same improvement.
    
    y_adjusted = y - f_best
    score = -y_adjusted / cost
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Fallback to simple exploitation: -mu, adjusted slightly for cost if needed
         # But simple -mu is safer for validity
         score = -mu / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Optional: Add a small cost penalty to break ties more strongly
    # But the 1/cost factor already does this.
    
    return score
