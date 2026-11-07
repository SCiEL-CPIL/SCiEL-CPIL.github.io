import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    Uses a Thompson Sampling-inspired stochastic offset combined with adaptive LCB.
    """
    # Track time to avoid timeout
    t_start = time.time()
    deadline = t_start + 285.0  # safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma to be non-negative and finite
    sigma = np.clip(sigma, 0.0, None)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)

    # Ackley 2D: min 0, max ~29
    f_max_ackley = 29.0
    f_ratio = np.clip(f_best / f_max_ackley, 0.0, 1.0)

    # Adaptive beta: start exploratory, become exploitative
    # Use a range that encourages more exploration early
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_ratio

    # Compute base LCB score: -mu + beta * sigma
    # Higher is better (lower predicted value or higher uncertainty)
    lcb_score = -mu + beta * sigma

    # Thompson Sampling-inspired stochastic offset
    # Sample a scalar offset from N(f_best, sigma_offset)
    # where sigma_offset scales with the maximum uncertainty in the candidate set
    max_sigma = np.max(sigma) if sigma.size > 0 else 0.0
    if max_sigma < 1e-8:
        max_sigma = 1e-8
    
    # The offset standard deviation should be meaningful relative to the scale of the problem
    # Use a fraction of max_sigma to create a stochastic shift
    # This effectively shifts the objective: we maximize -(mu - offset) / cost
    # which is equivalent to (offset - mu) / cost
    # By sampling offset ~ N(f_best, sigma_offset), we stochastically balance exploration/exploitation
    
    sigma_offset = 0.5 * max_sigma
    offset = np.random.normal(f_best, sigma_offset)
    
    # Thompson score: (offset - mu) / cost
    # This is equivalent to -mu/cost + offset/cost
    # The offset term is constant across all candidates, so it doesn't affect argmax
    # But we combine it with the LCB approach for a hybrid
    
    # Hybrid approach: combine LCB score with a Thompson-inspired penalty
    # The key insight from Thompson Sampling is that we should consider the probability
    # that a point is better than the current best, which relates to the CDF of the GP posterior
    
    # For each candidate, compute the probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is a principled way to balance exploration and exploitation
    
    # Avoid division by zero in the z-score
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values
    z = np.clip(z, -10, 10)
    
    # Compute probability using normal CDF
    # Use a stable approximation or scipy if available, but we'll use numpy
    # P(f(x) < f_best) = 0.5 * (1 + erf(z / sqrt(2)))
    prob_better = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    
    # Cost-aware scoring:
    # We want to maximize the probability of improvement per unit cost
    # But also incorporate the LCB uncertainty bonus
    
    # Combine: score = (alpha * prob_better + (1-alpha) * lcb_normalized) / cost
    # where lcb_normalized is the LCB score scaled to [0, 1]
    
    # Normalize lcb_score to [0, 1]
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range < 1e-8:
        lcb_range = 1e-8
    lcb_normalized = (lcb_score - lcb_min) / lcb_range
    
    # Blend probability-based and LCB-based scores
    # Use a fixed blend that favors the probability-based approach
    alpha = 0.7
    blended = alpha * prob_better + (1.0 - alpha) * lcb_normalized
    
    # Apply cost-aware normalization
    # Use 1/sqrt(cost) to penalize expensive points but not too aggressively
    cost_weight = 1.0 / np.sqrt(cost)
    
    score = blended * cost_weight
    
    # Check deadline
    if time.time() - t_start > deadline:
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
