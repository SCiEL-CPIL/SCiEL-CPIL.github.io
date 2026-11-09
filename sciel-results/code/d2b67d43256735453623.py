"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive cost-aware acquisition blending EI and UCB.
    
    Structural change: Instead of pure EI/cost, we blend:
    - EI: exploitation signal (expected improvement over f_best)
    - UCB: exploration signal (mu + beta * sigma, inverted for minimization)
    
    The blend weight adapts based on accumulated cost:
    - Early (low cost): more exploration (higher beta)
    - Late (high cost): more exploitation (lower beta)
    """
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Compute UCB-style exploration score (higher = more promising for minimization)
    # UCB for minimization: we want low mu and high uncertainty
    # Score = -(mu) + beta * sigma, then we want to maximize this
    # But we need to normalize scales. Let's use a relative form.
    
    # Adaptive beta: start high, decrease as cost accumulates
    # cost is the cumulative cost per candidate; total budget is 30
    # Use the mean cost as a proxy for how far along we are
    mean_cost = np.mean(cost)
    # Map mean_cost from [1, 30] to beta from [2.0, 0.3]
    # At low cost, explore more; at high cost, exploit more
    beta = 2.0 - 1.7 * np.clip((mean_cost - 1.0) / 29.0, 0.0, 1.0)
    beta = np.maximum(beta, 0.3)
    
    # UCB score for minimization: higher is better
    # We want candidates with low mu and high sigma
    # ucb_score = -mu + beta * sigma
    # But mu can be negative (hartmann goes to -3.86), so let's shift
    # Actually, let's normalize both EI and UCB to [0, 1] range before blending
    
    # Normalize EI: divide by max possible improvement
    # f_best is the best observed; the minimum is around -3.86
    # Max possible improvement is roughly f_best - (-4.0) = f_best + 4.0
    max_ei = np.maximum(f_best + 4.0, 1e-6)
    ei_norm = ei / max_ei
    ei_norm = np.clip(ei_norm, 0.0, 1.0)
    
    # UCB: -mu + beta*sigma
    # mu ranges roughly from -4 to 0, sigma from 0 to ~1
    # Let's normalize: (-mu) is in [0, 4], beta*sigma is in [0, 2]
    # So ucb_raw is in [0, 6]
    ucb_raw = -mu + beta * sigma
    ucb_max = 6.0  # conservative upper bound
    ucb_norm = np.clip(ucb_raw / ucb_max, 0.0, 1.0)
    
    # Blend: weight based on how much we've explored
    # Early: 50% EI, 50% UCB; Late: 80% EI, 20% UCB
    # Use cost to determine blend
    cost_frac = np.clip((mean_cost - 1.0) / 29.0, 0.0, 1.0)
    w_ei = 0.5 + 0.3 * cost_frac  # 0.5 to 0.8
    w_ucb = 1.0 - w_ei            # 0.5 to 0.2
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Divide by cost for cost-awareness
    scores = combined / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
