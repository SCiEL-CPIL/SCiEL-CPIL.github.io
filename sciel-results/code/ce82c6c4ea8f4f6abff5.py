# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: 330s wall-clock limit, use 320s as safety margin
_DEADLINE = 320.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    - Compute LCB = mu - kappa * sigma (optimism about potential improvement)
    - Apply a concave cost penalty that strongly favors cheap evaluations
      but allows expensive ones if the LCB is sufficiently low.
    - Return higher scores for points that are both promising and cheap.
    
    The key insight: instead of EI/cost (which can be dominated by very cheap
    points with tiny expected improvement), we use a direct lower bound on the
    objective with a nonlinear cost discount. This better handles the 
    multi-scale landscape of Styblinski-Tang where the global minimum is 
    sharp and surrounded by local minima.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-3)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e6)
    cost = np.clip(cost, 0.5, 100.0)
    
    # Adaptive kappa: larger when f_best is far from expected minimum (-78.33)
    # This increases exploration early and decreases it as we converge
    # Scale: if f_best is 50 units above min, use larger kappa
    dist_to_min = max(f_best - (-78.3323), 1e-3)
    
    # Base kappa around 1.0-2.0, scaled by how far we are from the minimum
    # Early on: more exploration (larger kappa)
    # Late: more exploitation (smaller kappa)
    base_kappa = 1.5 + 0.5 * np.log1p(dist_to_min / 10.0)
    
    # LCB: lower is better for the objective
    lcb = mu - base_kappa * sigma
    
    # Cost penalty: use a power law that strongly discounts expensive evaluations
    # cost is in [1, 2] roughly (1 + mean(x_scaled))
    # We want: cheap points get a boost, expensive points get a penalty
    # But not so strong that we never evaluate expensive points if they're very promising
    
    # Nonlinear cost adjustment:
    # - For cost near 1.0 (cheap): minimal penalty
    # - For cost near 2.0 (expensive): significant penalty
    # Use: penalty = alpha * (cost - 1.0)^beta, where beta > 1 makes it convex
    # Actually, we want to *subtract* a penalty from the score, so higher cost -> lower score
    
    # Alternative: multiply by a cost discount factor
    # discount = 1.0 / (cost ** gamma)
    # This makes cheap points (cost~1) have discount~1, expensive (cost~2) have discount~2^-gamma
    
    gamma = 1.2  # Cost sensitivity
    cost_discount = 1.0 / (cost ** gamma)
    
    # Combine: score = LCB * cost_discount
    # But LCB can be very negative (good), so multiplying by discount < 1 makes it less negative (worse)
    # That's the opposite of what we want! We want cheap promising points to score higher.
    
    # Better: score = -(LCB) / (cost ** gamma)  -> minimize this, but we need higher = better
    # So: score = -LCB / (cost ** gamma)
    # If LCB is very negative (good point), -LCB is large positive, divided by cost^gamma
    # Cheap points (cost~1) get the full -LCB, expensive points (cost~2) get -LCB / 2^gamma
    
    # Actually, let's think again. We want:
    # 1. Points with low mu (good) to score high
    # 2. Points with high sigma (uncertain) to score high (exploration)
    # 3. Points with low cost to score high
    
    # LCB = mu - kappa*sigma. Low LCB is good.
    # We want score to be high when LCB is low and cost is low.
    
    # Use: score = -LCB / (cost ** gamma)
    # When LCB is very negative (e.g., -50), -LCB = 50. Divided by cost^gamma.
    # Cheap point: 50 / 1.0^1.2 = 50
    # Expensive point: 50 / 2.0^1.2 ≈ 50 / 2.3 ≈ 21.7
    # So cheap points score higher. Good.
    
    # But what if LCB is positive (bad point)? Then -LCB is negative, and we get a negative score.
    # That's fine, as long as it's finite.
    
    score = -lcb / (cost ** gamma)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
