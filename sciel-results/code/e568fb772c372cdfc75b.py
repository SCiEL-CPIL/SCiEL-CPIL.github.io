"""Rewritten program: LCM-periodic systematic search to pack distinct multiples."""
import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0

    # Initial solution: the divisors themselves
    best_assign = divisors[:]
    best_val = max(best_assign) - min(best_assign)  # 33

    # Precompute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    global_lcm = reduce(lcm, divisors)  # 30030

    # Search for optimal window width W
    # Theoretical minimum is 9 (10 distinct integers)
    for W in range(9, best_val + 1):
        if time.time() > deadline:
            break
        
        # Due to periodicity, we only need to check L in [0, global_lcm + W - 1]
        # However, checking every L is expensive.
        # Optimization: For a fixed W, the set of valid Ls is periodic with period global_lcm.
        # We can iterate L from 0 to global_lcm + W.
        
        found = False
        # We can be smarter: we only care about L such that there is at least one multiple for each d.
        # Let's just iterate L. 30030 * 10 (max W) is ~300k iterations, very fast.
        for L in range(0, global_lcm + W):
            if time.time() > deadline:
                return [int(v) for v in best_assign]
            
            candidates = []
            valid = True
            for d in divisors:
                k1 = (L + d - 1) // d  # ceil(L/d)
                k2 = (L + W) // d       # floor((L+W)/d)
                if k1 > k2:
                    valid = False
                    break
                candidates.append((k1, k2))
            
            if not valid:
                continue
            
            # Try to assign distinct values using backtracking
            # Sort by number of candidates (most constrained first)
            order = sorted(range(n), key=lambda i: candidates[i][1] - candidates[i][0])
            used = set()
            assignment = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return assignment[:]
                i = order[idx]
                d = divisors[i]
                k1, k2 = candidates[i]
                for k in range(k1, k2 + 1):
                    val = k * d
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        result = backtrack(idx + 1)
                        if result is not None:
                            return result
                        used.remove(val)
                return None
            
            result = backtrack(0)
            if result is not None:
                # Verify span is exactly W or less? 
                # The problem asks to minimize max-min. 
                # If we find a solution in window [L, L+W], the span is <= W.
                # Since we iterate W from min, the first valid W is optimal.
                best_val = W
                best_assign = result[:]
                found = True
                break
        
        if found:
            break

    # Final validation
    result = [int(v) for v in best_assign]
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int)
        assert not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n
    
    return result
