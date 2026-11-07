import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with a cost-aware penalty.
    EI provides a principled exploitation signal. We add a cost penalty
    to discourage evaluating expensive points, but keep it mild to allow
    exploration of the full domain (where minima might be shifted).
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)

    # Calculate Expected Improvement (EI)
    # Standard EI: E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # Note: For minimization, we want to minimize f(x).
    # Improvement is f_best - f(x).
    # Let Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    from scipy.stats import norm
    
    Z = (f_best - mu) / sigma
    
    # Calculate Phi and phi
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (due to numerical issues it might be slightly negative)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score:
    # We want to maximize EI but penalize high cost.
    # A common approach is EI - beta * cost or EI / (1 + beta * cost).
    # Given the cost range is [1, 2] (since x in [-2, 2] scaled to [0, 1], 
    # cost = 1 + mean(x_scaled) is in [1, 2]), the penalty should be mild.
    # If we use EI / cost, a point with EI=1 and cost=2 scores 0.5.
    # A point with EI=0.9 and cost=1 scores 0.9.
    # This strongly favors cheap points.
    # However, the minimum might be in a high-cost region (shifted).
    # Let's try a softer penalty: EI / (1 + 0.1 * cost) or similar.
    # Or simply EI - 0.05 * cost.
    # Let's stick to a ratio but with a smaller denominator effect.
    # Score = EI / (1 + cost^0.5) ? No, cost is already normalized-ish.
    
    # Let's use: Score = EI - lambda * cost
    # If EI is in objective units (Rosenbrock values can be large),
    # and cost is in [1, 2], lambda needs to be small.
    # But EI depends on the scale of the problem.
    # In the early stages, f_best might be large (e.g., 100). EI could be large.
    # In later stages, f_best is small. EI is small.
    # A fixed lambda might not work well across scales.
    
    # Alternative: Cost-aware EI defined as EI / cost^alpha.
    # With alpha=0.5, it's a mild penalty.
    # Let's use EI / np.sqrt(safe_cost)
    # This is similar to the Thompson Sampling approach but uses the deterministic EI.
    
    scores = EI / np.sqrt(safe_cost)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
