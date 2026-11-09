"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for the deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 325 seconds to be safe against the 330s kill switch

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Combines a high-coefficient UCB (for aggressive exploration of distant minima)
    with a Probability of Improvement (POI) term (for exploitation of known good regions).
    The blend is adaptive: early on, UCB dominates to find new basins; late on,
    POI dominates to refine the best found so far.
    
    Parameters:
    mu: numpy array of shape (N,), posterior mean
    sigma: numpy array of shape (N,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    numpy array of shape (N,), acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a deterministic fallback: pure exploitation
        return -mu
    
    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. UCB component for minimization:
    # High beta to aggressively explore high-variance regions (find distant minima)
    # Branin is multimodal; standard UCB with low beta often gets stuck.
    beta = 5.0
    ucb_scores = beta * sigma_safe - mu
    
    # 2. POI component:
    # Probability that f(x) < f_best.
    # This provides a soft exploitation signal that is robust to scale.
    z_poi = (f_best - mu) / sigma_safe
    poi_scores = norm.cdf(z_poi)
    
    # Normalize both components to [0, 1] for stable blending
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    ucb_norm = _normalize(ucb_scores)
    poi_norm = _normalize(poi_scores)
    
    # 3. Adaptive blending:
    # We want to shift from exploration (UCB) to exploitation (POI) as we improve.
    # f_best starts high (e.g., ~100) and drops to ~0.4.
    # We use the gap between f_best and the known global minimum (0.397887) to gauge progress.
    # If f_best is far from min, we are early -> weight UCB higher.
    # If f_best is close to min, we are late -> weight POI higher.
    
    f_min_global = 0.397887
    remaining = max(f_best - f_min_global, 0.0)
    
    # Smooth transition:
    # When remaining is large (early), w_ucb is high.
    # When remaining is small (late), w_ucb is low.
    # Use an exponential decay.
    # At start, remaining might be ~50-100. exp(-50/20) is small.
    # Let's tune the scale. If remaining=10, exp(-0.5)=0.6.
    # If remaining=1, exp(-0.05)=0.95.
    # We want w_ucb to go from ~0.8 (early) to ~0.2 (late).
    # Let's try: w_ucb = 0.2 + 0.6 * exp(-remaining / 10.0)
    # If remaining=100, exp(-10) ~ 0 -> w_ucb = 0.2. This is too low for early exploration.
    # Branin has 7 local minima. We need strong exploration early.
    # Let's make the decay slower or the base higher.
    # Actually, with only 30 evaluations total, "late" is very late.
    # Eval 1-5: Random.
    # Eval 6-30: 25 steps.
    # f_best drops quickly in first few steps.
    # Let's use a step-based heuristic if possible? No, only f_best is passed.
    # Let's rely on f_best.
    
    # If f_best > 10, we are definitely early.
    # If f_best < 2, we are late.
    
    # Let's try a simpler, more aggressive exploration weight early.
    # w_ucb = 0.8 if f_best > 5 else 0.5 if f_best > 1 else 0.3
    # Smooth version:
    # w_ucb = 0.3 + 0.5 * exp(-f_best / 5.0)
    # If f_best=100, exp(-20)~0, w_ucb=0.3. Still low.
    # The problem is f_best starts very high.
    # Let's just use a fixed high exploration weight that decays slightly.
    # Or, simpler: UCB with high beta is already very exploratory.
    # POI is very exploitative.
    # A 50/50 blend might be too exploitative.
    # Let's weight UCB more heavily.
    
    # Let's try: w_ucb = 0.7, w_poi = 0.3
    # But adapt slightly: if f_best is very low, increase POI weight.
    
    # Heuristic:
    # If f_best > 10: w_ucb = 0.8, w_poi = 0.2
    # If f_best < 1:  w_ucb = 0.4, w_poi = 0.6
    # Interpolate smoothly.
    
    # Map f_best range [1, 10] to w_ucb range [0.4, 0.8]
    # Clamp f_best to [1, 10]
    f_clamped = np.clip(f_best, 1.0, 10.0)
    # Linear interpolation
    # t = (10 - f_clamped) / 9.0  (0 when f=10, 1 when f=1)
    t = (10.0 - f_clamped) / 9.0
    w_ucb = 0.8 - 0.4 * t  # 0.8 to 0.4
    w_poi = 1.0 - w_ucb    # 0.2 to 0.6
    
    score = w_ucb * ucb_norm + w_poi * poi_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
