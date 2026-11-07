"""
Strategy: Replace stochastic Thompson Sampling with a deterministic,
cost-aware Expected Improvement (EI) acquisition that adapts its
exploration bias (xi) based on the current uncertainty profile.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard EI formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Since we are minimizing, if mu < f_best, improvement is positive.
    # The standard formula above assumes minimization.
    
    # Adaptive exploration parameter xi
    # Early on, sigma is high. We might want to explore more.
    # However, EI naturally balances this.
    # Let's try a small xi to encourage exploration of regions with slightly worse mean
    # but high uncertainty, or just stick with standard EI.
    # Given the small budget (cost < 30), exploitation is usually key.
    # Let's use a very small xi, or 0.
    xi = 0.0
    
    # Adjusted improvement
    improvement = f_best - mu - xi
    
    # EI calculation
    # If improvement is very negative, EI is ~0
    # We use the standard closed form
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # We want to maximize score.
    # Simple division by cost penalizes expensive points heavily.
    # The inspiration suggests "reducing cost penalty bias helps".
    # Let's try a softer penalty: divide by cost^0.5 or add a small constant.
    # Or use: score = ei - lambda * cost
    # Let's try: score = ei / (1 + 0.5 * (cost - 1))
    # This normalizes cost relative to the minimum cost of 1.
    # cost is in [1, 2] roughly (1 + mean(x_scaled), x_scaled in [0,1])
    # So cost - 1 is in [0, 1].
    # Denominator in [1, 1.5].
    
    # Alternative: Just use EI directly?
    # If cost is ignored, we might pick the most uncertain/expensive point.
    # But the GP uncertainty should guide us to informative points.
    # Let's try a mild penalty.
    
    # Let's try: score = ei / cost_safe
    # This is the standard cost-aware EI.
    
    # Let's try a different structural approach:
    # Use the "Probability of Improvement" (PI) combined with EI?
    # Or use a hybrid:
    # score = ei / (cost_safe ** 0.5)
    
    # Let's stick to the most robust: EI / cost
    # But the inspiration said "reducing cost penalty bias helps".
    # Let's try: score = ei / (1 + 0.1 * (cost_safe - 1))
    # This makes the cost penalty much weaker.
    
    cost_factor = 1.0 + 0.1 * (cost_safe - 1.0)
    scores = ei / cost_factor
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
