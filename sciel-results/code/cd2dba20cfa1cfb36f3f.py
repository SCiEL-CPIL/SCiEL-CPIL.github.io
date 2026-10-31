"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import gamma

# Global state for adaptive learning
_start_time = None

# Weibull parameter estimates
# We will estimate shape k and scale lambda from the stream
# Initial guesses
_est_k = 2.0
_est_lambda = 15.0
_item_count = 0
_sum_items = 0.0
_sum_sq_items = 0.0

# Deadline guard
_DEADLINE = 120.0

def _estimate_weibull_params(mean, std):
    """
    Estimate Weibull shape (k) and scale (lambda) from mean and std.
    Uses the coefficient of variation (CV) relationship.
    For Weibull:
    Mean = lambda * Gamma(1 + 1/k)
    Var = lambda^2 * (Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)
    CV = std / mean = sqrt(Var) / Mean
    
    We can approximate k from CV.
    For k=2, CV = sqrt(Gamma(2) - Gamma(1.5)^2) / Gamma(1.5) = sqrt(1 - 0.785) / 0.886 = 0.464 / 0.886 = 0.52
    For k=1 (Exp), CV = 1.0
    For k=3, CV is lower.
    
    Simple approximation: k ≈ 1 / (0.5 * CV^2 + 0.2) ? No, let's use a lookup or simple inverse.
    Actually, a common approximation is k ≈ 3.6 - 6*(std/mean)^2 is for log-normal?
    Let's use a more robust method: 
    If CV is close to 1, k is close to 1.
    If CV is small, k is large.
    
    We can solve for k using Newton's method or a simple polynomial fit.
    Let's use a simple fit: k = 1 / (0.6 * CV^2 + 0.1) is rough.
    Better: 
    CV^2 = Gamma(1+2/k) / Gamma(1+1/k)^2 - 1
    Let g(k) = Gamma(1+2/k) / Gamma(1+1/k)^2 - 1
    We want g(k) = CV^2.
    
    Since we are in a tight loop, we use a pre-computed lookup or a very fast approximation.
    Approximation: k ≈ 1 / (0.55 * CV^2 + 0.15)
    Let's verify:
    k=1, CV=1.0 -> 1 / (0.55+0.15) = 1/0.7 = 1.4 (a bit off)
    k=2, CV=0.52 -> CV^2=0.27 -> 1 / (0.55*0.27+0.15) = 1 / (0.148+0.15) = 1/0.298 = 3.3 (off)
    
    Let's just use a fixed k=2.0 if estimation is too slow/unstable, but adapt lambda.
    Or, simpler: Just adapt Lambda based on Mean, assuming k is fixed at a reasonable value like 1.5 or 2.0.
    The "dead space" shape depends on k, but the location depends on lambda.
    Adapting lambda is the most critical part.
    
    Let's fix k=2.0 for stability and speed, and adapt lambda = mean / Gamma(1+1/k).
    """
    if std < 1e-6:
        return 2.0, mean
    cv = std / mean
    # Use a simple inversion table or approximation
    # For Weibull, CV decreases as k increases.
    # k=1.0 -> CV=1.0
    # k=2.0 -> CV=0.52
    # k=3.0 -> CV=0.36
    # k=4.0 -> CV=0.28
    
    # Linear interpolation on log scale?
    # Let's just clamp k between 1.0 and 4.0 based on CV.
    if cv >= 1.0:
        k_est = 1.0
    elif cv <= 0.25:
        k_est = 4.0
    else:
        # Rough linear fit between (1.0, 1.0) and (4.0, 0.25)
        # k = 1.0 + (1.0 - cv) * (3.0 / 0.75) = 1 + 4*(1-cv)
        k_est = 1.0 + 4.0 * (1.0 - cv)
        k_est = np.clip(k_est, 1.0, 4.0)
        
    lambda_est = mean / gamma(1.0 + 1.0/k_est)
    return k_est, lambda_est

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Adaptive distribution-aware strategy.
    """
    global _start_time, _est_k, _est_lambda, _item_count, _sum_items, _sum_sq_items, _DEADLINE

    if _start_time is None:
        _start_time = time.time()

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback to simple Best Fit
        remainders = bins - item
        valid_mask = remainders >= 0
        scores = np.full_like(bins, -1e9, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remainders[valid_mask]
        return scores

    # Update running statistics
    _item_count += 1
    _sum_items += item
    _sum_sq_items += item * item
    
    # Calculate mean and std
    mean = _sum_items / _item_count
    variance = max(_sum_sq_items / _item_count - mean * mean, 1e-6)
    std = np.sqrt(variance)
    
    # Update Weibull parameters
    _est_k, _est_lambda = _estimate_weibull_params(mean, std)
    
    capacity = 100.0
    remainders = bins - item
    valid_mask = remainders >= 0

    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    valid_remainders = remainders[valid_mask]
    
    # Distribution-aware scoring
    # We want to minimize the "Expected Dead Space" of the remainder r.
    # Dead space is high if r is large but unlikely to be filled by future items.
    # Probability that a future item x > r is S(r) = exp(-(r/lambda)^k).
    # If S(r) is high, most items are too big to fit in r. So r is dead.
    # The "waste" is roughly r * S(r).
    # However, if r is very small, r * S(r) is small, but we lose consolidation.
    # If r is very large, S(r) is small, so r * S(r) is small.
    # The function f(r) = r * exp(-(r/lambda)^k) has a peak.
    # We want to AVOID the peak.
    # So we want to minimize f(r) + consolidation_bias.
    # Consolidation bias: we prefer smaller r generally, but not if it's in the dead zone.
    # Actually, minimizing f(r) pushes us to r=0 or r=inf.
    # In bin packing, r=0 is Best Fit. r=inf is Worst Fit.
    # The shape of f(r) determines the balance.
    # If lambda is small (items are small), the peak is at small r. We should avoid small r? 
    # No, if items are small, S(r) drops off quickly. 
    # Let's check: lambda=10, k=2. Peak at r ~ lambda * k^(-1/k) ~ 10 * 2^(-0.5) ~ 7.
    # So remainders around 7 are "dead" because items > 7 are common? 
    # S(7) = exp(-(0.7)^2) = exp(-0.49) = 0.61. 61% of items are > 7. 
    # So a remainder of 7 is likely to be wasted by the next item (which is probably > 7).
    # So we should avoid leaving remainders around 7.
    # We should prefer remainders < 7 (tight fit, might fit small items) or > 7 (loose fit, can fit big items).
    # But usually, tight fit (Best Fit) is better for consolidation.
    # So we penalize the "dead zone" specifically.
    
    # Calculate survival function S(r)
    scaled = valid_remainders / _est_lambda
    # Handle potential overflow/underflow
    survival = np.exp(-np.power(scaled, _est_k))
    
    # Dead space metric
    dead_space = valid_remainders * survival
    
    # We want to MINIMIZE dead_space.
    # But we also want to consolidate (minimize remainder).
    # If we just minimize dead_space, we might pick a huge remainder (Worst Fit) if the dead zone is in the middle.
    # However, in bin packing, opening new bins is bad. 
    # The "dead space" of a large remainder is low (because S(r) is low), but it wastes capacity.
    # We need a penalty for large remainders too.
    # Let's add a linear penalty for the remainder itself, but scaled down.
    # penalty = dead_space + alpha * remainder
    
    # If alpha is small, we mostly care about dead space.
    # If alpha is large, we mostly care about consolidation.
    # Let's use alpha = 0.1
    penalty = dead_space + 0.1 * valid_remainders
    
    # Score is negative penalty (higher is better)
    scores_valid = -penalty
    
    # Small tie-breaker
    scores_valid -= np.arange(len(valid_remainders)) * 1e-10
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
