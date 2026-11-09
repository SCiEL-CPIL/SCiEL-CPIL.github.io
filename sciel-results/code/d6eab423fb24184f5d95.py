"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = [None]
_call_count = [0]

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using quantile-based cost-normalized improvement probability.
    
    Strategy:
    1. Compute Probability of Improvement (PI) for each candidate.
    2. Normalize PI by cost to get a 'cost-efficiency' score.
    3. Use a dynamic threshold based on the quantile of PI to filter out 
       low-confidence candidates in early stages, prioritizing exploitation 
       of known good regions (which are often cheaper if we guide the search).
    4. In later stages, shift weight towards pure cost-normalized PI to ensure 
       we squeeze the last bit of value from the budget.
    """
    if _start_time[0] is None:
        _start_time[0] = time.time()
    elapsed = time.time() - _start_time[0]
    if elapsed > 325:
        # Safe fallback: prefer low mu, low cost
        mu_safe = np.where(np.isfinite(mu), mu, 0.0)
        cost_safe = np.where(np.isfinite(cost), cost, 1.0)
        return -mu_safe / (cost_safe + 1e-9)
    
    _call_count[0] += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.5)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10) # Clip to avoid numerical issues, though norm.cdf handles it
    pi = norm.cdf(z)
    
    # Cost-normalized PI
    # We want to maximize the probability of improvement per unit cost
    cost_pi = pi / (cost + 1e-9)
    
    # Discrete Quantile Strategy:
    # Instead of a smooth blend, we create a "gate".
    # If the PI is very low (below a certain quantile), we heavily penalize the score
    # unless the cost is extremely low. This forces the algorithm to either find a 
    # point with decent confidence of improvement OR a very cheap point to explore.
    
    n_calls = _call_count[0]
    # Progress estimate: 5 initial + ~15 more = 20 total evals approx
    # frac goes from 0 to 1
    frac = min(1.0, n_calls / 20.0)
    
    # Dynamic quantile threshold for PI
    # Early on (frac=0), we want to be open to exploration, so threshold is low (e.g., 10th percentile)
    # Late on (frac=1), we want to be strict, so threshold is high (e.g., 90th percentile)
    # This "quantile gate" effectively creates a discrete set of preferred candidates.
    q_low = 0.1
    q_high = 0.9
    q_threshold = q_low + (q_high - q_low) * frac
    
    # Compute the threshold value for PI
    pi_threshold = np.quantile(pi, q_threshold)
    
    # Create a binary mask: 1 if PI >= threshold, 0 otherwise
    # We use a soft mask to allow some flexibility but strongly prefer passing the gate
    # If PI is below threshold, we multiply the score by a small factor (0.1)
    # If PI is above threshold, we keep the full score.
    gate = np.where(pi >= pi_threshold, 1.0, 0.1)
    
    # Additionally, we want to boost low-cost candidates if they pass the gate
    # or if they are extremely cheap (cost < 1.2)
    cheap_boost = np.where(cost < 1.2, 1.5, 1.0)
    
    # Final score: Cost-normalized PI * Gate * Cheap Boost
    # We also add a small exploitation term based on -mu for tie-breaking among high PI points
    # But the primary driver is the cost-normalized probability.
    
    # To make it more "discrete" and robust to noise, we can bin the scores.
    # However, continuous scores are required for the return. 
    # The "discrete" aspect comes from the Gate and the Quantile threshold.
    
    scores = cost_pi * gate * cheap_boost
    
    # Add a small exploitation component to break ties and guide towards lower mu
    # scaled by the gate so we don't exploit low PI regions
    exploit_term = -mu * 0.1 * gate
    
    scores += exploit_term
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
