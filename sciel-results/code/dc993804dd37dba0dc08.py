# EVOLVE-BLOCK-START
import time
import random

def _valid_set(s, N):
    """Return True iff no element of s is the arithmetic mean of any
    subset (size >=2) of the OTHER elements.  Uses integer bitsets."""
    n = len(s)
    if n < 3:
        return True
    others = [0] * n
    for i in range(n):
        m = (1 << n) - 1
        m &= ~(1 << i)
        others[i] = m
    for i in range(n):
        a = s[i]
        others_mask = others[i]
        # dp[k] = bitset of achievable sums of exactly k elements from others[i]
        dp = [0] * (n + 1)
        dp[0] = 1
        m = others_mask
        while m:
            low = m & (-m)
            j = (low.bit_length() - 1)
            m ^= low
            v = s[j]
            for k in range(n, 0, -1):
                dp[k] |= dp[k - 1] << v
        for k in range(2, n):
            if (dp[k] >> (k * a)) & 1:
                return False
    return True

def _is_2_valid_addition(s, x):
    """Check if adding x to s maintains the property that no element is the mean of a subset of size 2."""
    # 1. Check if x is the mean of two elements in s
    # 2x = a + b
    # We can check this by iterating through s? No, use a set for O(1) lookup if we had sums.
    # Since s is small, O(n^2) is fine, but we can do better.
    # Actually, just check if 2x is in the set of pair sums of s.
    # But we don't have precomputed sums.
    # For O(n) check:
    # If 2x = a+b, then b = 2x - a.
    # We need b in s, b != a, b != x (x not in s yet).
    s_set = set(s)
    for a in s:
        b = 2 * x - a
        if b in s_set and b != a:
            return False
    
    # 2. Check if any a in s becomes the mean of a pair involving x
    # a = (b + x) / 2  =>  2a = b + x  =>  b = 2a - x
    # b must be in s, b != a, b != x.
    for a in s:
        b = 2 * a - x
        if b in s_set and b != a and b != x:
            return False
            
    return True

def construct():
    t0 = time.time()
    DEADLINE = t0 + 50.0  # Safety margin
    N = 512
    MAXC = 48
    rng = random.Random(20240607)

    # Phase 1: Greedy construction with O(n) heuristic (checks size 2 means only)
    # We try to add numbers in a random order to avoid bias
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    best = []
    for x in candidates:
        if len(best) >= MAXC:
            break
        if _is_2_valid_addition(best, x):
            best.append(x)
            
    best.sort()
    current_len = len(best)
    
    # Phase 2: Verify and Repair
    # The greedy set is 2-valid, but might fail for size >= 3.
    # We check full validity. If invalid, we remove elements causing violations.
    
    def find_violation(s):
        """Returns (i, subset_indices) if s[i] is mean of subset, else None."""
        n = len(s)
        for i in range(n):
            a = s[i]
            others = [j for j in range(n) if j != i]
            # Use DP to find if k*a is achievable by k others
            # We only need to find ONE violation
            dp = [0] * (n + 1)
            dp[0] = 1
            for j in others:
                v = s[j]
                for k in range(n, 0, -1):
                    dp[k] |= dp[k - 1] << v
            for k in range(2, n):
                if (dp[k] >> (k * a)) & 1:
                    # Found violation for s[i] with subset size k
                    # We don't easily have the subset indices from the bitset DP here
                    # But we know i is the culprit
                    return i
        return None

    # Repair loop
    while time.time() < DEADLINE:
        if _valid_set(best, N):
            break
        
        v_idx = find_violation(best)
        if v_idx is None:
            break
            
        # Remove the element causing the violation
        # Or remove a random element from the potential subset? 
        # Simplest: remove v_idx
        best.pop(v_idx)
        best.sort()
        
        # If we removed too many, we might stop improving
        if len(best) < 40:
             # Try a different removal strategy? 
             # For now, just continue
             pass

    # If we still have time, try to add more numbers?
    # The greedy phase might have skipped good numbers due to 2-validity constraints 
    # that were too strict or due to random order.
    # Let's do a second greedy pass with the remaining candidates?
    # Actually, the current best is likely close to optimal if 2-validity is the main bottleneck.
    
    return best

# EVOLVE-BLOCK-END
