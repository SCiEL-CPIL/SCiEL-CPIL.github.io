import time
import math
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    def generate_sums_with_history(weights):
        """Generate all sums and store history of reachable sets for backtracking."""
        history = [set([0])]
        curr = {0}
        for wi in weights:
            next_set = set()
            for s in curr:
                next_set.add(s - wi)
                next_set.add(s)
                next_set.add(s + wi)
            curr = next_set
            history.append(curr)
        return list(curr), history
    
    def backtrack_coeffs(weights, history, target_sum):
        """Backtrack to find coefficients that achieve target_sum."""
        coeffs = [0] * len(weights)
        current_sum = target_sum
        for i in range(len(weights) - 1, -1, -1):
            wi = weights[i]
            # Try +wi, 0, -wi in order
            if current_sum - wi in history[i]:
                coeffs[i] = 1
                current_sum -= wi
            elif current_sum in history[i]:
                coeffs[i] = 0
            elif current_sum + wi in history[i]:
                coeffs[i] = -1
                current_sum += wi
            else:
                # Should not happen if target_sum is reachable
                coeffs[i] = 0
        return coeffs
    
    # Generate sums with history for backtracking
    sums1, hist1 = generate_sums_with_history(w1)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums2, hist2 = generate_sums_with_history(w2)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums1.sort()
    
    best_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    # For each sum in sums2, binary search in sums1 for the closest to -s2
    for s2 in sums2:
        target = -s2
        idx = bisect.bisect_left(sums1, target)
        
        # Check idx and idx-1 (and a few neighbors for safety)
        candidates = set()
        for check_idx in [idx, idx - 1, idx + 1, idx - 2, idx + 2]:
            if 0 <= check_idx < len(sums1):
                candidates.add(sums1[check_idx])
        
        for s1 in candidates:
            total = s1 + s2
            if total != 0 and abs(total) < best_val:
                best_val = abs(total)
                best_s1 = s1
                best_s2 = s2
    
    if time.time() > deadline or best_s1 is None:
        return [0] * 28 + [1, -1]
    
    coeffs1 = backtrack_coeffs(w1, hist1, best_s1)
    coeffs2 = backtrack_coeffs(w2, hist2, best_s2)
    
    result = coeffs1 + coeffs2
    
    # Verify the result
    total = Fraction(0, 1)
    for i, c in enumerate(result):
        if c != 0:
            total += Fraction(c, i + 1)
    
    if total == 0:
        # Fallback: find any nonzero combination
        return [0] * 28 + [1, -1]
    
    return result
