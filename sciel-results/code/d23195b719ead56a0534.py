import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement per unit cost with dynamic penalty
    # z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost penalty:
    # - Use log(cost) to softly penalize expensive points
    # - Scale the penalty by a factor that decays as f_best approaches 0
    # - When f_best is large (early), penalty is small -> explore expensive regions
    # - When f_best is small (late), penalty is large -> exploit cheap points
    
    # f_best starts around the function value at random points, which for powell 4d
    # on [-4,5]^4 can be quite large. As we find better points, f_best decreases toward 0.
    
    # Adaptive penalty coefficient:
    # - Use a sigmoid-like transition based on f_best
    # - When f_best > threshold_high, penalty is minimal
    # - When f_best < threshold_low, penalty is maximal
    
    # Powell 4d minimum is 0. Typical initial f_best might be in range [10, 100+].
    # Let's use a smooth transition centered around f_best = 1.0
    
    # penalty_weight = 1 / (1 + exp(-k * (f_best - center)))
    # When f_best >> center: penalty_weight -> 1 (strong penalty)
    # When f_best << center: penalty_weight -> 0 (weak penalty)
    
    # Actually, we want the OPPOSITE: strong penalty when f_best is small (close to optimum)
    # So: penalty_weight = 1 / (1 + exp(k * (f_best - center)))
    # When f_best >> center: penalty_weight -> 0 (weak penalty, explore)
    # When f_best << center: penalty_weight -> 1 (strong penalty, exploit cheap)
    
    k = 2.0
    center = 1.0
    penalty_weight = 1.0 / (1.0 + np.exp(k * (f_best - center)))
    
    # Also add a base penalty that's always present to prevent pure cost-ignoring
    base_penalty = 0.3
    
    # Total penalty coefficient
    penalty_coeff = base_penalty + penalty_weight
    
    # Score = EI - penalty_coeff * log(cost)
    # This rewards high EI and penalizes high cost, with the penalty strength adapting
    
    log_cost = np.log(cost)
    score = ei - penalty_coeff * log_cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
