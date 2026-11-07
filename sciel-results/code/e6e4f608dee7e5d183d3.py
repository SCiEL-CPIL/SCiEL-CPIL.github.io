"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 310.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a knowledge-gradient-style value-of-information
    score. The value of evaluating a point is estimated as the expected reduction
    in simple regret (f_best) if that point is the true best, weighted by the
    probability it actually improves, divided by acquisition cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = better candidate to evaluate)
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score: how many std devs below f_best is the predicted mean
    z = (f_best - mu) / sigma_safe
    
    # Probability that this point is better than current best
    # P(f(x) < f_best) = Phi(z)
    prob_improve = norm.cdf(z)
    
    # Expected value of f(x) given it's below f_best (truncated normal mean)
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z)/Phi(z)
    # The improvement over f_best if we evaluate and it improves:
    # E[max(0, f_best - f(x))] = EI (standard expected improvement)
    phi_z = norm.pdf(z)
    # Guard against division by zero when prob_improve is tiny
    prob_safe = np.maximum(prob_improve, 1e-15)
    truncated_mean_improvement = sigma_safe * phi_z / prob_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * prob_improve + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Knowledge gradient approximation:
    # The value of information is proportional to:
    #   - How much we expect to improve (EI)
    #   - The uncertainty (sigma) — more uncertainty means more to learn
    #   - The probability of improvement — if very unlikely, less value
    #
    # KG-style score: EI * (1 + alpha * sigma / (f_best - mu + eps))
    # But we want to be cost-aware.
    #
    # Alternative structural approach: use the "probability-weighted EI"
    # which is EI itself, but modulate by an uncertainty bonus that
    # decays with cost.
    
    # Cost normalization: map cost [1,2] to a penalty weight
    # Use a power law: cost^gamma where gamma > 1 for super-linear penalty
    # gamma = 1.5 makes high-cost points penalized more aggressively
    gamma = 1.5
    cost_penalty = np.power(np.maximum(cost, 1.0), gamma)
    
    # Base score: EI
    base_score = ei
    
    # Uncertainty bonus: points with high sigma have higher value of information
    # even if EI is moderate. Scale by 1/sqrt(cost) so expensive points
    # get less exploration bonus.
    sigma_bonus = sigma_safe / np.sqrt(cost_penalty)
    
    # Weight the uncertainty bonus
    alpha = 0.15
    
    scores = (base_score + alpha * sigma_bonus) / cost_penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
