import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 320s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 320:
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost
    
    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    pi_val = norm.cdf(z_clipped)
    
    # Expected Improvement (EI)
    pdf_val = norm.pdf(z_clipped)
    ei_val = (f_best - mu) * pi_val + safe_sigma * pdf_val
    
    # Normalize EI by the scale of the problem
    # Use a robust scale based on the spread of f_best and sigma
    scale_factor = max(abs(f_best), np.mean(safe_sigma) + eps)
    ei_norm = ei_val / scale_factor
    
    # Structural Change: Use a different combination strategy
    # Instead of a simple linear blend, use a multiplicative gate
    # This emphasizes exploitation when confidence is high (high PI)
    # and allows exploration when confidence is low.
    # The term (1 - pi_val) acts as an exploration bonus that decays as confidence grows.
    # We combine EI (normalized) with a term that rewards high sigma (exploration)
    # but is gated by the probability of NOT improving (i.e., uncertainty).
    
    # Let's try a "Thompson Sampling-like" score in expectation
    # Score = E[max(0, f_best - y)] / cost  (which is EI)
    # But we want to encourage exploration.
    # Alternative: Use UCB (Upper Confidence Bound) for minimization
    # UCB = mu + kappa * sigma
    # We want to minimize, so we want low mu and high sigma.
    # Acquisition for minimization UCB: score = -(mu - kappa * sigma) / cost
    # This is equivalent to (kappa * sigma - mu) / cost
    
    # Let's blend EI and UCB
    # EI is bounded and smooth. UCB is linear in sigma.
    # For griewank, which is multimodal but smooth, EI is usually good.
    # However, the current score is 1.5461, which is decent.
    # The directive asks for a STRUCTURAL change.
    # Let's try a pure UCB approach with an adaptive kappa.
    # Or a hybrid that switches based on the number of evaluations (estimated by cost sum?)
    # We don't have the number of evaluations, but we have the cost array.
    # The cost array has shape (2000,), representing the cost of evaluating each candidate?
    # No, "cost" has shape (2000,). "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # So cost[i] is the cost of evaluating candidate i.
    # The total cost is the sum of costs of evaluated points.
    # We don't have access to the history of evaluated points in the acquisition function signature.
    # We only have mu, sigma, f_best, cost for the 2000 candidates.
    
    # Let's try a different structural change:
    # Use a "Probability of Improvement" based acquisition but with a penalty for cost
    # that is more aggressive, or a reward for exploration.
    
    # Let's try: Score = (EI + beta * sigma) / cost
    # This adds a direct exploration bonus.
    # Normalize sigma to be comparable to EI.
    sigma_norm = safe_sigma / scale_factor
    beta = 0.5  # Exploration weight
    
    combined_score = ei_norm + beta * sigma_norm
    
    # Cost-aware
    score = combined_score / safe_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
