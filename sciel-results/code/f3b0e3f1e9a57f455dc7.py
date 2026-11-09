"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma has a small floor
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Stochastic knapsack approach:
    # Value density = P(f < f_best) * E[improvement | f < f_best] / cost
    # P(f < f_best) = Phi(z) where z = (f_best - mu)/sigma
    # E[improvement | f < f_best] = (f_best - mu) + sigma * phi(z)/Phi(z)
    # This is equivalent to EI / Phi(z) * Phi(z) = EI, but the probabilistic framing
    # suggests we should weight by the probability of improvement explicitly.
    
    # Probability of improvement
    pi_prob = norm.cdf(z)
    
    # Conditional expected improvement given improvement occurs
    # E[f_best - f | f < f_best] = (f_best - mu) + sigma * phi(z) / Phi(z)
    # But we need to handle the case where pi_prob is very small
    safe_pi = np.maximum(pi_prob, 1e-10)
    phi_z = norm.pdf(z)
    conditional_ei = (f_best - mu) + sigma_safe * phi_z / safe_pi
    conditional_ei = np.maximum(conditional_ei, 0.0)
    
    # Value density: probability-weighted improvement per unit cost
    # This is essentially EI / cost, but let's be more explicit about the probabilistic efficiency
    cost_base = np.maximum(cost, 1e-6)
    
    # Primary score: EI / cost (expected improvement per unit cost)
    primary_score = ei / cost_base
    
    # Secondary score: probability-weighted conditional improvement per unit cost
    secondary_score = pi_prob * conditional_ei / cost_base
    
    # Blend: use a weighted combination
    # The primary score (EI/cost) is standard; the secondary emphasizes high-probability improvements
    # Let's also add a UCB-like exploration term to help escape local minima
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        # UCB term: beta * sigma / cost
        beta_ucb = 0.1
        ucb_term = beta_ucb * sigma_safe / cost_base
    else:
        ucb_term = 0.0
    
    # Combine: primary + secondary contribution + UCB
    # The secondary score is already pi * conditional_ei / cost = ei / cost = primary
    # So they're equivalent. Let's use a different approach.
    
    # Alternative: use the probability of improvement as a gate
    # High probability of improvement + low cost = high priority
    # But also reward high uncertainty regions for exploration
    
    # Let's try: score = (ei + alpha * sigma) / cost^gamma
    # where alpha controls exploration and gamma controls cost sensitivity
    
    # From the knowledge: UCB with cost penalty (exploration-heavy) worked better
    # score = (beta * sigma + ei) / cost
    
    # Let's use a balanced approach:
    # score = (ei + beta * sigma) / cost^0.9
    # The 0.9 exponent is slightly sublinear to not over-penalize cost
    
    cost_penalized = cost_base ** 0.9
    beta_ucb = 0.2  # Exploration weight
    scores = (ei + beta_ucb * sigma_safe) / cost_penalized
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
