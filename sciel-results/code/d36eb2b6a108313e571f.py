"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters estimated for the specific distribution regime
# Capacity 100, 1000 items.
K = 2.0
LAMBDA = 16.0
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy:
    Model the bin packing process as a discrete queueing system.
    The score for each bin is the negative of the expected waiting time for the next item 
    to fit into the resulting remainder.
    
    Let r be the remaining capacity after placing the item.
    The next item X is drawn from Weibull(K, LAMBDA).
    We want to estimate the expected number of items that will be skipped (or the time) 
    before an item fits into space r.
    
    The probability that a single item fits is P(X <= r) = F(r).
    The expected number of trials until success (geometric distribution) is 1/F(r).
    This is a proxy for "waiting time" or "waste".
    
    However, simply minimizing 1/F(r) is equivalent to maximizing F(r), which is monotonic 
    with r. This would favor large r (Worst-Fit), which is generally bad.
    
    Why? Because if r is very large, F(r) is close to 1, so waiting time is ~1. 
    But we waste a lot of space.
    
    Let's refine the "waiting time" concept. 
    Instead of just counting items, consider the "dead space" created.
    
    Alternative Queueing Model:
    Think of the bin as a server with capacity r.
    Items arrive. If they fit, they are served. If not, they are lost (to other bins).
    The "cost" of a bin state is related to how much capacity is idle.
    
    Let's look at Inspiration 1 and 2. They use "dead space" metrics.
    Inspiration 2 (score 1.8171) minimizes the "dead zone" probability.
    
    Let's try a different structural change:
    Instead of a single-step expected waste, let's look at the **hazard rate** of the Weibull distribution.
    
    The hazard rate h(r) = f(r) / S(r) represents the instantaneous probability of an item arriving that fits exactly into space r (or is small enough, in a differential sense).
    Actually, for a gap r, the probability density that the next item is size x (where x <= r) is f(x).
    
    Let's try scoring based on the **expected utilization** of the gap r by the next item, 
    but penalized by the **variance** or **risk** of the gap being wasted.
    
    Actually, the prompt suggests: "negative of the expected waiting time for the next item to fit into the resulting remainder, calculated as the ratio of the remainder to the Weibull hazard rate at that size."
    
    Let's interpret this literally first.
    Hazard rate of Weibull at x: h(x) = (k/lambda) * (x/lambda)^(k-1)
    At size r: h(r) = (k/lambda) * (r/lambda)^(k-1)
    
    Score = - r / h(r) ?
    Score = - r / [(k/lambda) * (r/lambda)^(k-1)]
    Score = - (r * lambda / k) * (lambda/r)^(k-1)
    Score = - (lambda / k) * r * (lambda/r)^(k-1)
    Score = - (lambda^k / k) * r^(2-k)
    
    If k=2, Score = - (lambda^2 / 2) * r^(0) = Constant.
    This is useless for k=2.
    
    If k < 2, Score increases with r (Worst-Fit).
    If k > 2, Score decreases with r (Best-Fit).
    
    For Weibull with k=2 (Rayleigh-like), the hazard rate is linear.
    
    Let's rethink the "waiting time" metaphor.
    Maybe it's not about the item fitting *exactly* at r, but fitting *into* r.
    
    Let's try a different structural approach: **Lookahead Heuristic**.
    Instead of just the next item, consider the expected fill of the bin over the next few items.
    
    However, we must be fast.
    
    Let's look at the successful Inspiration 2: "minimizing the 'dead zone' probability".
    Dead zone: A gap r is a "dead zone" if the probability of the next item fitting is low, OR if the expected waste is high.
    
    Let's try a **Hybrid** approach that is structurally different from the current "Expected Waste" model.
    Current: Score = E[X * 1(X<=r)] - r.
    
    New Idea: **Relative Fit Score**.
    Score = (Probability Next Item Fits) * (Expected Fraction of Gap Used | Fits)
    
    P(Fits) = F(r)
    E[Fraction | Fits] = E[X | X<=r] / r
    
    Score = F(r) * (E[X | X<=r] / r)
    
    Let's analyze this.
    If r is very small: F(r) is small. E[X|X<=r] is small.
    If r is very large: F(r) is ~1. E[X|X<=r] is ~E[X]. Score ~ E[X]/r -> 0.
    
    This score peaks at some intermediate r!
    This is a "Sweet Spot" heuristic. It avoids both very small gaps (low probability of fit) and very large gaps (low utilization).
    
    Let's calculate E[X | X<=r].
    E[X | X<=r] = E[X * 1(X<=r)] / F(r)
    
    So Score = F(r) * ( (E[X * 1(X<=r)] / F(r)) / r )
    Score = E[X * 1(X<=r)] / r
    
    This is exactly the "Expected Fraction of Gap Used by Next Item".
    
    Let's check the behavior:
    r -> 0: E[X * 1(X<=r)] ~ ∫0^r x f(x) dx.
    f(x) ~ x^(k-1). x f(x) ~ x^k. Integral ~ r^(k+1).
    Score ~ r^(k+1) / r = r^k -> 0.
    
    r -> ∞: E[X * 1(X<=r)] -> E[X].
    Score ~ E[X] / r -> 0.
    
    So the score is 0 at r=0 and r=∞, and positive in between.
    This favors intermediate gaps.
    
    Is this good?
    Best-Fit favors small r. Worst-Fit favors large r.
    This favors "Good-Fit" or "Next-Fit Decreasing" like behavior?
    
    For k=2, lambda=16:
    E[X] = lambda * Gamma(1+1/2) = 16 * Gamma(1.5) = 16 * 0.5 * sqrt(pi) ≈ 16 * 0.886 ≈ 14.2.
    
    If r = 14.2 (mean size):
    Score = E[X * 1(X<=14.2)] / 14.2.
    Since 14.2 is the mean, roughly half the mass is below.
    E[X * 1(X<=14.2)] is maybe 60% of total E[X]? ~8.5.
    Score ≈ 8.5 / 14.2 ≈ 0.6.
    
    If r = 5 (small):
    F(5) is small. E[X * 1(X<=5)] is very small.
    Score is small.
    
    If r = 50 (large):
    F(50) is high. E[X * 1(X<=50)] ≈ E[X] ≈ 14.2.
    Score ≈ 14.2 / 50 ≈ 0.28.
    
    So r=14.2 (Score 0.6) is better than r=50 (Score 0.28) and r=5 (Score <0.1).
    
    This suggests we should try to fill bins to a level where the remaining gap is around the mean item size.
    This is a "Target Gap" heuristic.
    
    This is structurally different from "Minimize Expected Waste" (which favors r=0) and "Maximize Probability of Fit" (which favors r=∞).
    
    Let's implement this "Expected Fraction of Gap Used" score.
    
    Score(r) = I(r) / r
    where I(r) = E[X * 1(X<=r)] = lambda * Gamma(1+1/k) * P(1+1/k, (r/lambda)^k)
    
    We need to handle r=0. If r=0, score is 0 (or undefined, but limit is 0).
    
    Let's add a small epsilon to avoid division by zero if r is extremely small but >0.
    
    Also, we should consider the number of bins. If we have many bins, maybe we want to spread out?
    No, the task is to pick the best bin for the current item.
    
    Let's try this.
    
    One refinement: The "Expected Fraction of Gap Used" assumes the next item is the *only* consideration.
    But in bin packing, we care about the *total* number of bins.
    
    Let's stick to this simple, structurally different heuristic first.
    
    """
    scores = np.full(len(bins), -1e18)
    
    remaining = bins - item
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_r = remaining[valid_mask]
    
    # Avoid division by zero
    valid_r_safe = np.where(valid_r > 1e-9, valid_r, 1e-9)
    
    # Precompute constant
    try:
        from scipy.special import gammainc, gamma
        gamma_s = gamma(1.0 + 1.0/K)
    except:
        import math
        gamma_s = math.gamma(1.0 + 1.0/K)
    
    # Calculate (r/lambda)^k
    z = np.power(valid_r_safe / LAMBDA, K)
    
    # Calculate lower incomplete gamma ratio P(s, z)
    p_vals = gammainc(1.0 + 1.0/K, z)
    
    # I(r) = lambda * gamma(s) * P(s, z)
    expected_used = LAMBDA * gamma_s * p_vals
    
    # Score = Expected Fraction of Gap Used
    # We want to maximize this.
    # If r is very small, I(r) is very small, so Score is small.
    # If r is very large, I(r) is ~E[X], so Score is ~E[X]/r, which is small.
    # Peak is at intermediate r.
    
    scores[valid_indices] = expected_used / valid_r_safe
    
    return scores
# EVOLVE-BLOCK-END
