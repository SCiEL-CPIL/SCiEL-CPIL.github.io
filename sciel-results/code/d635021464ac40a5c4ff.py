"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0  # seconds
_start_time = None


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety: fall back to simple best fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores
    
    remaining = bins - item
    feasible = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
        
    valid_remaining = remaining[feasible]
    
    # Weibull parameters estimated for typical online bin packing tasks
    # with capacity 100.
    # k (shape) ~ 2.0, lambda (scale) ~ 15.0 
    _WEIBULL_K = 2.0
    _WEIBULL_LAMBDA = 15.0
    
    # Structural Idea from Inspiration 1: Distribution-aware "Expected Dead Space"
    # We model the waste of leaving a remainder `r` as the probability that a future item
    # drawn from the Weibull distribution will NOT fit, weighted by the remainder size.
    # This creates a landscape where:
    # 1. Very small remainders are good (low absolute waste, low chance of fitting anything).
    # 2. Medium remainders are bad (high chance of fitting an item, significant absolute waste if it doesn't).
    # 3. Very large remainders are penalized less by the "fit probability" but still have high absolute waste.
    
    # Survival Function S(r) = P(X > r) = exp(-(r/lambda)^k)
    # We want to MINIMIZE the expected dead space.
    # A proxy for "Dead Space" is r * S(r) or just S(r) depending on interpretation.
    # If we use r * S(r), we penalize large r even if unlikely to be filled.
    # If we use S(r), we only care about the probability of being filled.
    
    # Let's use a composite score that encourages "Best Fit" behavior but modulated by the distribution.
    # The inspiration suggests: Score = - (CDF(r) * r) or similar.
    # Actually, the insight "minimizing the 'dead zone' probability" suggests we want to avoid remainders
    # where P(fit) is high but we didn't fill it. But we ARE filling it now.
    # The remainder r is what's left.
    # If r is such that many future items fit (high CDF), we should be happy? No, because we are online.
    # If we leave r, and a future item x <= r arrives, it fits. Good.
    # If x > r, it doesn't fit. Bad (waste r).
    # So we want to maximize the chance that r is useful?
    # Or minimize the chance that r is useless?
    
    # Let's look at the "Adversarial Robustness" from Inspiration 1.
    # Minimize "Regret".
    # Regret of leaving r is the probability that a future item DOES NOT fit? No, that's just waste.
    # If we leave r, and no future item fits, r is dead.
    # If we leave r, and a future item fits, r is used.
    # We want to maximize the probability that r is used by SOME future item?
    # This is hard to model exactly.
    
    # Let's stick to the specific formula hinted at in the inspiration's code comments:
    # "Score = - (1 - CDF(r)) * r" -> Penalizes large r that are unlikely to be filled?
    # Wait, if r is large, 1-CDF(r) is small. So penalty is small.
    # If r is medium, 1-CDF(r) is medium. Penalty is medium.
    # If r is small, 1-CDF(r) is ~1. Penalty is ~r (small).
    # This function is low at r=0, rises to a peak, then falls to 0 as r->inf.
    # Minimizing this penalty means preferring r=0 or r->inf.
    # This encourages either perfect fits (Best Fit) or Worst Fit.
    # In bin packing, Best Fit is generally superior to Worst Fit for consolidation.
    # However, pure Best Fit can be sensitive to noise.
    
    # Let's try the "Expected Dead Space" metric:
    # Waste(r) = E[max(0, r - X)] = r - E[min(X, r)]
    # We want to minimize Waste(r).
    # For Weibull, E[min(X, r)] = integral_0^r S(t) dt.
    # Approximation: If r is small, S(t) ~ 1, so E[min(X,r)] ~ r. Waste ~ 0.
    # If r is large, S(t) decays.
    
    # Let's use a simpler, computationally efficient surrogate that captures the "distribution-aware best fit" spirit.
    # Score = - (remaining * exp(-(remaining/lambda)^k))
    # This is the "Dead Space" metric: r * S(r).
    # We want to MINIMIZE this. So Score = - (r * S(r)).
    
    # Let's check the shape of f(r) = r * exp(-(r/lambda)^k).
    # f(0) = 0.
    # f'(r) = exp(...) + r * exp(...) * (-k/lambda) * (r/lambda)^(k-1)
    # f'(r) = exp(-(r/lambda)^k) * (1 - (k * r^k) / lambda^k)
    # Critical point at r = lambda * (1/k)^(1/k).
    # For k=2, lambda=15: r* = 15 * (0.5)^0.5 = 15 * 0.707 = 10.6.
    # So the "Dead Space" is maximized around r=10.6.
    # We want to AVOID this region.
    # We want r to be close to 0 (Best Fit) or very large (Worst Fit).
    # Since Best Fit is usually better, we should add a term that breaks the tie in favor of smaller r.
    # Or, we can just use -r as a secondary term.
    
    # Composite Score:
    # 1. Primary: - (r * S(r))  -> Minimizes expected dead space.
    # 2. Secondary: - r         -> Breaks ties by preferring smaller remainders (Best Fit bias).
    # 3. Tertiary: - r^2        -> Stronger penalty for large gaps to ensure consolidation.
    
    # Let's refine the weights.
    # r * S(r) is in the range [0, ~10].
    # r is in [0, 100].
    # We need to scale them.
    
    # Let's use the log-sum-exp soft-min idea from the current program?
    # No, the current program's soft-min was on the set of bins, which is stateful/global per call.
    # The inspiration is stateless per bin.
    
    # Let's try:
    # score = - (valid_remaining * np.exp(-(valid_remaining / _WEIBULL_LAMBDA) ** _WEIBULL_K))
    # This creates a valley at r=0 and r->inf.
    # To ensure we pick the BEST bin (smallest r among those with low dead space), we add -r.
    
    # Let's combine:
    # dead_space = valid_remaining * np.exp(-(valid_remaining / _WEIBULL_LAMBDA) ** _WEIBULL_K)
    # best_fit_penalty = valid_remaining
    
    # We want to minimize (dead_space + alpha * best_fit_penalty).
    # So Score = - (dead_space + alpha * best_fit_penalty).
    
    # What is alpha?
    # If alpha is very large, it's pure Best Fit.
    # If alpha is 0, it's pure Dead Space minimization (which might pick large r).
    # We want a bias towards Best Fit.
    # Let's set alpha = 0.1.
    
    # Let's also consider the "Polarization" from the current program.
    # Polarization encouraged bins to be full or empty.
    # The Dead Space metric already encourages full (r=0) or empty (r=100).
    # But "empty" (r=100) is not good for packing efficiency if we have many items.
    # However, if we have a bin with r=100, it's a new bin. We shouldn't prefer it over a bin with r=10 if r=10 is "dead".
    # Wait, if r=10 is dead (high dead space), and r=100 is not dead (low dead space), we prefer r=100.
    # This means we open a new bin instead of using the one with r=10.
    # This is actually a known strategy: "Best Fit Decreasing" vs "Worst Fit".
    # But in online packing, opening new bins is costly.
    # We should penalize large r more strongly to avoid opening new bins unnecessarily.
    # The term -r does this.
    
    # Let's try:
    # Score = - (dead_space + 0.5 * valid_remaining)
    
    # Let's test the logic:
    # Bin A: r = 5. dead_space = 5 * exp(-(5/15)^2) = 5 * exp(-0.11) = 5 * 0.89 = 4.45.
    #            score = - (4.45 + 0.5 * 5) = - (4.45 + 2.5) = -6.95.
    # Bin B: r = 10. dead_space = 10 * exp(-(10/15)^2) = 10 * exp(-0.44) = 10 * 0.64 = 6.4.
    #            score = - (6.4 + 0.5 * 10) = - (6.4 + 5.0) = -11.4.
    # Bin C: r = 50. dead_space = 50 * exp(-(50/15)^2) = 50 * exp(-11.1) = 50 * 1.5e-5 = 0.00075.
    #            score = - (0.00075 + 0.5 * 50) = - (0.00075 + 25) = -25.
    # Bin D: r = 1. dead_space = 1 * exp(-(1/15)^2) = 1 * exp(-0.0044) = 0.995.
    #            score = - (0.995 + 0.5 * 1) = -1.495.
    
    # Ranking: D (-1.495) > A (-6.95) > B (-11.4) > C (-25).
    # This prefers r=1 (Best Fit).
    # What if r=10 was the only option vs r=50?
    # B (-11.4) > C (-25). Prefers r=10.
    # What if r=15 vs r=50?
    # r=15: dead_space = 15 * exp(-(15/15)^2) = 15 * exp(-1) = 15 * 0.368 = 5.52.
    #        score = - (5.52 + 0.5 * 15) = - (5.52 + 7.5) = -13.02.
    # r=50: score = -25.
    # Prefers r=15.
    
    # This looks like a robust Best Fit variant.
    # The "Dead Space" term modulates the penalty.
    # For small r, dead_space ~ r. Score ~ - (r + 0.5r) = -1.5r.
    # For medium r (around 10), dead_space is higher. Score is lower.
    # For large r, dead_space is low, but 0.5r is high. Score is low.
    
    # This effectively creates a smooth "Best Fit" curve that is steeper in the "dead zone".
    
    # Let's add a small quadratic term to ensure that for very large r, the penalty grows faster than linearly?
    # The linear term -0.5r already does that.
    
    # Let's refine the Weibull parameters.
    # If items are Weibull(k=2, lambda=15), mean is ~13.
    # If items are smaller, lambda should be smaller.
    # Let's stick with lambda=15, k=2.
    
    # Calculate scores for valid bins
    dead_space = valid_remaining * np.exp(-(valid_remaining / _WEIBULL_LAMBDA) ** _WEIBULL_K)
    best_fit_term = valid_remaining
    
    # Composite score
    valid_scores = - (dead_space + 0.5 * best_fit_term)
    
    scores[feasible] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
