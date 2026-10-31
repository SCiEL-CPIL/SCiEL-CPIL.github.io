import numpy as np
import time
from scipy.stats import weibull_min

# Global state
_start_time = None

# Weibull distribution parameters
# Based on the successful island 2 finding, we use a distribution-aware metric.
# The parameters k=2.0 and lambda=15.0 were part of the successful program p333.
# However, the exact parameters might be tuned. Let's stick to the ones that worked.
# Mean = lambda * Gamma(1 + 1/k)
# For k=2, lambda=15: Mean = 15 * Gamma(1.5) = 15 * 0.8862 = 13.29
# The capacity is 100.
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Structural Change: Distribution-Aware Expected Dead Space
    # Instead of a heuristic "dead zone" based on a fixed ENI, we calculate
    # the expected probability that a future item will fit into the remaining space.
    # If the remaining space is `r`, the probability that the next item `X` fits is P(X <= r).
    # The "Expected Dead Space" contribution of this bin is roughly `r * (1 - P(X <= r))`?
    # Or more simply, we want to maximize the probability that the space is used.
    # Let's look at the successful program p333's logic:
    # "Replaces heuristic 'dead zone' estimation with a distribution-aware 'Expected Dead Space' metric (r * S(r))"
    # where S(r) is likely the survival function P(X > r).
    # So, Dead Space = r * P(X > r).
    # We want to MINIMIZE this dead space.
    # Therefore, Score = - (r * P(X > r)).
    
    # Let's verify this logic.
    # If r is very small (e.g., 1), and items are Weibull(2, 15), P(X > 1) is high (~0.9).
    # Dead Space = 1 * 0.9 = 0.9.
    # If r is 0, Dead Space = 0.
    # If r is 10, P(X > 10) is lower. Let's say 0.5. Dead Space = 5.0.
    # If r is 50, P(X > 50) is very low. Let's say 0.1. Dead Space = 5.0.
    # This suggests that large gaps are also bad if they are unlikely to be filled.
    # However, pure Best Fit (minimizing r) is often good because it consolidates.
    # The metric r * S(r) balances the size of the gap with the likelihood of it being wasted.
    
    # Let's implement this.
    # We need the survival function of the Weibull distribution.
    # S(r) = 1 - CDF(r) = exp(-(r / lambda)^k)
    
    # We calculate this for all valid remaining capacities.
    # r = new_rems
    # S(r) = np.exp(-(new_rems / _WEIBULL_LAMBDA) ** _WEIBULL_K)
    
    # Note: new_rems can be 0. (0/lambda)^k = 0. exp(0) = 1. Dead Space = 0 * 1 = 0. Good.
    # Note: new_rems can be large. S(r) -> 0. Dead Space -> 0. 
    # This implies that very large gaps are "good" because they are likely to be filled by SOME item eventually?
    # No, if S(r) is small, it means P(X > r) is small, so P(X <= r) is large.
    # Wait. S(r) = P(X > r).
    # If r is large, S(r) is small.
    # Dead Space = r * S(r).
    # If r=100, S(100) is tiny. Dead Space is tiny.
    # This would suggest Worst Fit (largest gap) is best?
    # That contradicts Best Fit performance.
    
    # Let's re-read the cell finding: "Expected Dead Space metric (r * S(r)) that mathematically identifies remainders likely to be wasted".
    # If S(r) = P(X > r), then r * S(r) is the expected amount of space r that is NOT filled by the NEXT item?
    # No. The expected value of min(X, r) is the expected space used.
    # The expected waste is r - E[min(X, r)].
    # E[min(X, r)] = integral_0^r P(X > t) dt.
    # So Expected Waste = r - integral_0^r S(t) dt.
    # This is more complex.
    
    # Let's look at the actual code of p333 if possible? No, I only have the description.
    # "r * S(r)" is a common approximation for tail risk.
    # Let's consider the alternative: We want to maximize the probability of filling the bin.
    # Score = P(X <= r) = 1 - S(r) = CDF(r).
    # This would prefer larger gaps (Worst Fit).
    
    # Let's reconsider the "Dead Space" concept.
    # If I have a gap of `r`, and the next item is `X`.
    # If X <= r, the gap becomes r - X.
    # If X > r, the gap remains r (and the item goes elsewhere).
    # The "dead" part of the gap is the part that will likely never be used.
    # A simple heuristic is that if r is small, it's likely dead. If r is large, it's likely alive.
    # So we want to AVOID small r.
    # Score should be HIGH for large r?
    # But Best Fit (small r) is usually better than Worst Fit (large r) for total bins.
    # Why? Because large bins fragment the "large" capacity.
    
    # Let's look at the score 1.817. It's a huge jump from 0.8196.
    # The parent p288 was 0.8196. The child p333 was 1.817.
    # The change was "Replaces heuristic 'dead zone' estimation with a distribution-aware 'Expected Dead Space' metric (r * S(r))".
    # And it shifted from "local gap minimization" to "global probabilistic waste minimization".
    
    # If the score is LOWER is better for the raw value, but the "search score" is seed/value?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # Wait. If raw value is lower is better, and seed value is V_seed.
    # Score = V_seed / V_child.
    # If V_child < V_seed, Score > 1.
    # So 1.817 means the child used significantly FEWER bins (lower raw excess) than the seed.
    
    # So the metric `r * S(r)` led to a BETTER packing (lower excess).
    # How?
    # If we minimize `r * S(r)`, what do we prefer?
    # Let's test values with Weibull(2, 15).
    # r=1: S(1) = exp(-(1/15)^2) = exp(-0.0044) = 0.9956. Score = 0.9956.
    # r=5: S(5) = exp(-(5/15)^2) = exp(-0.111) = 0.8948. Score = 4.474.
    # r=10: S(10) = exp(-(10/15)^2) = exp(-0.444) = 0.641. Score = 6.41.
    # r=20: S(20) = exp(-(20/15)^2) = exp(-1.77) = 0.170. Score = 3.40.
    # r=50: S(50) = exp(-(50/15)^2) = exp(-11.1) = 0.000015. Score = 0.00075.
    # r=100: S(100) = exp(-(100/15)^2) = exp(-44.4) ~ 0. Score = 0.
    
    # If we MINIMIZE this score (i.e., we want the bin with the lowest `r * S(r)`), we would prefer:
    # r=100 (0) > r=50 (0.00075) > r=20 (3.4) > r=10 (6.4) > r=5 (4.4) > r=1 (0.99).
    # This prefers LARGE gaps. This is Worst Fit.
    # Worst Fit is generally WORSE than Best Fit for bin packing.
    # Did the child use Worst Fit?
    # If the child scored 1.817 (better than seed), and the seed was likely Best Fit or similar, then Worst Fit must have worked better here?
    # Or maybe the score is MAXIMIZED?
    # "each arriving item goes to the highest-scoring bin".
    # If Score = - (r * S(r)), then we maximize - (r * S(r)), which is equivalent to minimizing (r * S(r)).
    # So it would pick the bin with the LARGEST gap (Worst Fit).
    
    # Is it possible that for Weibull sequences with capacity 100, Worst Fit is better?
    # Usually, Best Fit is robust. But maybe the specific Weibull parameters make "keeping bins open" (Worst Fit) better to catch large items?
    # Weibull has a long tail. Large items occur occasionally.
    # If you use Best Fit, you might fill up the "medium" bins and leave the "large" bins for the large items, but you fragment the medium bins.
    # If you use Worst Fit, you keep the large bins large, and fill the small bins? No, Worst Fit puts the item in the LARGEST bin.
    # So Worst Fit keeps the SMALL bins small? No, it keeps the LARGE bins large? No, it consumes the large bins.
    # Worst Fit: Item goes to bin with most space.
    # This tends to fill bins evenly? No, it tends to keep bins either full or empty?
    
    # Let's check the alternative interpretation.
    # Maybe the score is `S(r)`?
    # Maximize S(r) = P(X > r).
    # S(1) = 0.995. S(10) = 0.64. S(50) = 0.0001.
    # Maximizing S(r) prefers SMALL gaps (r=1). This is Best Fit.
    # Best Fit is a strong baseline.
    
    # Maybe the score is `1 - S(r)` = CDF(r)?
    # Maximize CDF(r). Prefers LARGE gaps. Worst Fit.
    
    # Let's look at the text again: "Expected Dead Space metric (r * S(r))".
    # If the program MINIMIZES the "Expected Dead Space", it minimizes `r * S(r)`.
    # As calculated, `r * S(r)` is low for large r and high for medium r.
    # So it prefers r -> infinity or r -> 0?
    # At r=0, score=0.
    # At r=1, score=1.
    # At r=100, score=0.
    # So it prefers r=0 (Perfect Fit) OR r=100 (Empty Bin).
    # It AVOIDS r=10 (Medium gap).
    # This is a "Bimodal" preference.
    # It wants bins to be either FULL or EMPTY.
    # This is a known heuristic! "Best Fit" approximates this by preferring full.
    # But if a perfect fit isn't available, Best Fit picks the smallest gap (e.g., r=5).
    # This new metric would prefer r=50 (score 0.00075) over r=5 (score 4.47).
    # So it would pick the LARGEST gap if no perfect fit exists?
    # Effectively, it acts like Worst Fit when perfect fits are rare, but strongly prefers perfect fits.
    
    # Given the huge score jump, this "Bimodal" strategy (avoid medium gaps) seems to be the key.
    # We will implement Score = - (r * S(r)).
    # This means we MAXIMIZE - (r * S(r)), which is MINIMIZING r * S(r).
    
    # Implementation details:
    # r = new_rems
    # S(r) = exp(-(r / lambda) ** k)
    # metric = r * S(r)
    # score = -metric
    
    # We also add a tiny tie-breaker.
    
    r = new_rems
    # Avoid division by zero or negative issues (r is always >= 0)
    # r can be 0.
    scaled_r = r / _WEIBULL_LAMBDA
    # Clip scaled_r to prevent overflow in exp if r is huge (though r <= 100)
    # (100/15)^2 = 44. exp(-44) is fine.
    survival_prob = np.exp(-(scaled_r ** _WEIBULL_K))
    
    expected_dead_space = r * survival_prob
    
    scores_valid = -expected_dead_space
    
    # Tie-breaking: prefer lower index? Or higher index?
    # Usually, index doesn't matter much, but consistency is good.
    # Let's add a tiny penalty for higher index to break ties deterministically.
    scores_valid -= np.arange(len(valid_indices)) * 1e-9
    
    scores[valid] = scores_valid
    
    return scores
