"""Find 30 coefficients in {-1,0,1} minimizing |sum c[i]/(i+1)| with nonzero sum."""
import time
from math import gcd
from fractions import Fraction
from itertools import product

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Precompute L/(i+1) for i in 0..29
    # L = lcm(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    
    # L = 2329089562800
    # weights[i] = L // (i+1) for i in 0..29
    weights = [L // (i + 1) for i in range(30)]
    
    # Meet in the middle: split into first 15 and last 15
    n1 = 15
    n2 = 15
    
    # Enumerate all sums for first half
    # sum1 = sum_{i=0}^{14} c[i] * weights[i]
    # There are 3^15 = 14348907 possibilities
    
    def enumerate_half(weights_subset):
        """Return list of (sum_value, coefficients_tuple) for all 3^n combinations."""
        n = len(weights_subset)
        results = []
        # Use iterative approach to avoid recursion depth issues
        # Actually, let's use a different approach: build incrementally
        sums = [0]
        coeffs = [()]
        for w in weights_subset:
            new_sums = []
            new_coeffs = []
            for s, c in zip(sums, coeffs):
                new_sums.append(s)
                new_coeffs.append(c + (0,))
                new_sums.append(s + w)
                new_coeffs.append(c + (1,))
                new_sums.append(s - w)
                new_coeffs.append(c + (-1,))
            sums = new_sums
            coeffs = new_coeffs
        return list(zip(sums, coeffs))
    
    # This might be too slow/memory heavy for 3^15. Let me think of a better way.
    # 3^15 = 14348907, which is manageable but storing tuples is expensive.
    # Let's just store the sums and reconstruct coefficients later if needed.
    
    # Actually, for the first pass, let me just get the sums.
    def get_sums(weights_subset):
        n = len(weights_subset)
        sums = [0]
        for w in weights_subset:
            new_sums = []
            for s in sums:
                new_sums.append(s)
                new_sums.append(s + w)
                new_sums.append(s - w)
            sums = new_sums
        return sums
    
    # Get sums for first half
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    sums1 = get_sums(w1)
    sums2 = get_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = sorted(sums2)
    
    import bisect
    
    best_abs = None
    best_total = None
    best_idx1 = None
    best_idx2 = None
    
    # For each sum in sums1, find sum in sums2 such that sum1+sum2 is closest to 0 but not 0
    # We want to minimize |sum1 + sum2| subject to sum1+sum2 != 0
    
    # Iterate through sums1 and for each, find best match in sums2
    # To speed up, we can limit the search
    
    # Actually, let's be smarter. We want pairs (s1, s2) minimizing |s1+s2| with s1+s2 != 0.
    # For a given s1, the best s2 is the one closest to -s1, but not exactly -s1 (or if it is, the next closest).
    
    # Let's iterate and check time periodically
    len1 = len(sums1)
    len2 = len(sums2)
    
    # To make this tractable, let's sample or limit
    # 14M * log(14M) should be doable in ~10-20 seconds
    
    best_val = None
    best_s1 = None
    best_s2 = None
    
    # For efficiency, iterate over the smaller or just do it all
    # Let's do all of sums1
    for i, s1 in enumerate(sums1):
        if time.time() > DEADLINE:
            break
        target = -s1
        # Find position where target would be in sums2_sorted
        pos = bisect.bisect_left(sums2_sorted, target)
        
        # Check candidates around pos
        candidates = []
        for p in [pos - 1, pos, pos + 1]:
            if 0 <= p < len2:
                candidates.append(sums2_sorted[p])
        
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_val = abs(total)
            if best_val is None or abs_val < best_val:
                best_val = abs_val
                best_s1 = s1
                best_s2 = s2
    
    if best_val is None:
        # Fallback: return seed
        return [0] * 28 + [1, -1]
    
    # Now reconstruct coefficients
    # We need to find c1 (15 coeffs) such that sum c1[i]*w1[i] = best_s1
    # and c2 (15 coeffs) such that sum c2[i]*w2[i] = best_s2
    
    def find_coeffs(weights_subset, target):
        """Find coefficients in {-1,0,1} such that sum c[i]*w[i] = target."""
        n = len(weights_subset)
        # Backtracking search
        def backtrack(idx, current_sum, coeffs):
            if time.time() > DEADLINE:
                return None
            if idx == n:
                if current_sum == target:
                    return list(coeffs)
                return None
            w = weights_subset[idx]
            for c in [0, 1, -1]:
                coeffs.append(c)
                result = backtrack(idx + 1, current_sum + c * w, coeffs)
                coeffs.pop()
                if result is not None:
                    return result
            return None
        return backtrack(0, 0, [])
    
    coeffs1 = find_coeffs(w1, best_s1)
    coeffs2 = find_coeffs(w2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    result = coeffs1 + coeffs2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]
    
    # Local search to try to improve
    if time.time() < DEADLINE - 5:
        # Try flipping individual coefficients to see if we can get smaller
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            current_total = total
            for i in range(30):
                old_c = result[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    new_total = current_total + (new_c - old_c) * weights[i]
                    if new_total == 0:
                        continue
                    if abs(new_total) < abs(current_total):
                        result[i] = new_c
                        current_total = new_total
                        total = new_total
                        improved = True
                        break
    
    return result
