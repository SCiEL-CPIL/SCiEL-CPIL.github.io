"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Explicit deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback: simple cost-aware PI
        m = np.asarray(mu, dtype=np.float64)
        s = np.asarray(sigma, dtype=np.float64)
        c = np.asarray(cost, dtype=np.float64)
        s = np.maximum(s, 1e-9)
        c = np.maximum(c, 1e-9)
        z = (m - f_best) / s
        pi = norm.cdf(z)
        pi = np.clip(pi, 0.0, 1.0)
        return np.where(np.isfinite(pi / c), pi / c, 0.0)
    
    # Convert to numpy arrays for efficiency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Probability of Improvement (PI) Component ---
    # PI is more robust to variance than EI. It measures the probability that a candidate
    # is better than the current best. For minimization, this is P(f(x) < f_best).
    # z = (mu - f_best) / sigma
    # PI = Phi(z)
    z = (mu - f_best) / sigma_safe
    # Clip z to avoid numerical issues, though norm.cdf handles large values well
    z_clipped = np.clip(z, -50, 50)
    pi = norm.cdf(z_clipped)
    
    # Cost-aware PI
    pi_cost = pi / cost_safe
    
    # --- Lower Confidence Bound (LCB) Component ---
    # LCB provides a deterministic exploration bonus.
    # For minimization, we want to minimize mu - beta * sigma.
    # So the score to maximize is -mu + beta * sigma.
    # Adaptive beta: scale with max uncertainty or a fixed value.
    # Let's use a fixed beta that balances exploration.
    # Since targets are standardized in the GP, the scale is roughly 1.
    # A beta of 1.0-2.0 is typical. Let's try 1.5.
    beta = 1.5
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware LCB
    lcb_cost = lcb_score / cost_safe
    
    # --- Blending ---
    # PI is stochastic in its interpretation (probability) but deterministic in calculation.
    # LCB is deterministic.
    # Let's blend them. PI values are in [0, 1]. LCB values can be negative or positive.
    # To blend, we should normalize them to similar scales.
    # Alternatively, we can just use a weighted sum.
    # Let's try 0.5 * PI + 0.5 * LCB.
    # However, LCB can be very large in magnitude compared to PI (which is bounded).
    # Let's normalize LCB by its standard deviation or max value to bring it to a similar scale.
    lcb_std = np.std(lcb_score)
    if lcb_std > 1e-9:
        lcb_normalized = lcb_score / lcb_std
    else:
        lcb_normalized = lcb_score
    
    # Now lcb_normalized is roughly in [-3, 3] range.
    # PI is in [0, 1].
    # Let's scale PI to have similar magnitude? Or just use weights.
    # If we use 0.5 * PI + 0.5 * LCB_normalized, the LCB part dominates due to range.
    # Let's try to make them comparable.
    # Actually, let's stick to the simpler approach:
    # score = w1 * (PI / cost) + w2 * (LCB / cost)
    # But LCB / cost can be very large.
    
    # Let's try a different structural change:
    # Use a cost-aware PI, but with a penalty for high cost.
    # score = PI * (1 / cost)
    # This is already pi_cost.
    
    # Let's try combining PI and LCB with careful scaling.
    # Normalize LCB to [0, 1] range? No, it's signed.
    # Let's use a sigmoid on LCB to bound it?
    # Or simply: score = PI / cost + alpha * (LCB / cost)
    # Let's try alpha = 0.1 to keep LCB as a secondary signal.
    
    scores = pi_cost + 0.1 * lcb_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
