import math
import time

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.95  # Use 95% of time budget

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols.
    2. Secondary: Maximize Shannon Entropy of the symbol distribution.
    3. Tertiary: Penalize vectors close to "dangerous corners" (all-0, all-1, all-2)
       using a quadratic distance metric. This addresses the structural risk of
       vectors that are part of many trivial APs or lie on low-dimensional subspaces.
    4. Quaternary: Robust hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > 300 * _DEADLINE_FRACTION:
        # Return a very low priority to stop processing if we are close to timeout
        # This is a safeguard, though the evaluator usually handles this.
        return -1e10

    # Count symbols
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c in counts if c > 0)
    
    # Shannon Entropy
    # H = - sum(p_i * log(p_i))
    # If p_i = 0, term is 0.
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    
    # Primary score: distinct count
    # Scale to make it dominant
    distinct_score = distinct_count * 10000.0
    
    # Secondary score: Higher entropy is better
    # Max entropy for n is log(3)
    # Scale by 1000 to make it significant but not overpower distinct count
    entropy_score = entropy * 1000.0
    
    # Tertiary score: Penalty for being close to dangerous corners
    # The "dangerous corners" are (0,0,...,0), (1,1,...,1), (2,2,...,2).
    # A vector close to these has a high concentration of one symbol.
    # We use the sum of squared deviations from the ideal uniform distribution (n/3).
    # This is equivalent to a chi-squared statistic (without the denominator) and
    # penalizes imbalance quadratically.
    ideal = n / 3.0
    dev0 = c0 - ideal
    dev1 = c1 - ideal
    dev2 = c2 - ideal
    imbalance_penalty = -(dev0*dev0 + dev1*dev1 + dev2*dev2)
    
    # Scale the imbalance penalty to be secondary to entropy but more sensitive than hash
    # The max possible value for (dev^2 sum) is when one count is n and others 0: 2*(n/3)^2 + (2n/3)^2 = 2n^2/3.
    # For n=10, this is ~66. We want this to have an impact but not overwhelm entropy (which is ~1.1).
    # Let's scale it by 10.
    imbalance_score = imbalance_penalty * 10.0
    
    # Quaternary: Robust hash
    # Convert vector to bytes for hashing
    # vector elements are 0, 1, 2
    # Use a simple polynomial hash with a large prime
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 13) % (10**9 + 7)
    
    # Normalize hash to [0, 1]
    hash_score = h / (10**9 + 7)
    
    return float(distinct_score + entropy_score + imbalance_score + hash_score)
