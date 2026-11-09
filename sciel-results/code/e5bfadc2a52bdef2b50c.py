"""Pack distinct multiples of a fixed integer set."""
import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    best_val = float('inf')
    best_result = None
    
    def try_window(L, W):
        candidate_sets = []
        for d in divisors:
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_min > k_max:
                return None
            candidates = [k * d for k in range(k_min, k_max + 1) if k * d <= MAX_V]
            if not candidates:
                return None
            candidate_sets.append(candidates)
        
        indices = sorted(range(n), key=lambda i: len(candidate_sets[i]))
        used = set()
        assignment = [None] * n
        
        def backtrack(pos):
            if pos == n:
                return True
            i = indices[pos]
            for val in candidate_sets[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(pos + 1):
                        return True
                    assignment[i] = None
                    used.remove(val)
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Search W from 9 upward
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        # For efficiency, check L values that are "aligned" with the divisors.
        # Key L values: multiples of each divisor, and multiples +/- small offsets.
        # Also, due to periodicity with lcm=30030, we only need L in [1, 30030].
        
        # Generate candidate L values
        candidate_Ls = set()
        for d in divisors:
            for offset in range(-2, W + 3):
                for base in range(0, 30030 + 1, d):
                    L = base + offset
                    if 1 <= L <= 30030:
                        candidate_Ls.add(L)
        
        # Also add some random samples
        import random
        random.seed(42)
        for _ in range(200):
            candidate_Ls.add(random.randint(1, 30030))
        
        found = False
        for L in sorted(candidate_Ls):
            if time.time() > deadline:
                break
            result = try_window(L, W)
            if result is not None:
                val = max(result) - min(result)
                if val < best_val:
                    best_val = val
                    best_result = result[:]
                found = True
                break
        
        if found:
            continue
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]
