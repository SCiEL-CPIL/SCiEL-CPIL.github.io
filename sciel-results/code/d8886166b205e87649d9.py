import time
from fractions import Fraction
from functools import lru_cache

def construct():
    deadline = time.time() + 85  # Safety margin within 90s
    
    # Try to find minimal-n solutions
    # For each n, do a bounded DFS
    
    def dfs(remaining, min_d, count, path):
        """Find 'count' more distinct denominators >= min_d that sum to 'remaining'."""
        if time.time() > deadline:
            return None
        if count == 0:
            if remaining == 0:
                return list(path)
            return None
        if remaining <= 0:
            return None
        
        # Pruning: max possible sum with 'count' terms starting from min_d
        # Each term <= 1/min_d, so max sum = count/min_d
        if remaining > Fraction(count, min_d):
            return None
        
        # Min possible: each term >= 1/10000000, so min sum = count/10000000
        # If remaining < count/10000000, impossible (but this is extremely small, rarely binds)
        
        # Try denominators from min_d upward
        # Upper bound for next denominator d: 1/d <= remaining, so d >= 1/remaining
        # Also need remaining - 1/d to be representable with (count-1) terms starting from d+1
        # Max sum of (count-1) terms from d+1: (count-1)/(d+1)
        # So need: remaining - 1/d <= (count-1)/(d+1)
        
        # Lower bound for d: 1/d <= remaining → d >= ceil(1/remaining)
        # But also d >= min_d
        
        # Upper bound for d: we need remaining - 1/d >= (count-1)/10000000 (very small)
        # And remaining - 1/d <= (count-1)/(d+1)
        # The binding constraint is usually the upper bound from the "can complete" check
        
        # Let's iterate d from min_d, with a reasonable upper bound
        # d_max: 1/d >= remaining/count (roughly, to not overshoot too much)
        # Actually d should be at least ceil(1/remaining) to not overshoot
        # And d should be such that we can still complete
        
        # Practical upper bound: try up to some limit
        d_start = max(min_d, 5)
        # d must satisfy: 1/d <= remaining → d >= 1/remaining
        # And: remaining - 1/d <= (count-1)/(d+1) for feasibility
        
        # Let's try a range of d values
        # The key insight: d can't be too large because then 1/d is tiny and we can't make up the sum
        # d can't be too small because then 1/d might exceed remaining
        
        # Upper bound on d: if d is very large, 1/d ≈ 0, and we need (count-1) terms to sum to ~remaining
        # But (count-1) terms each ≤ 1/(d+1) ≈ 0, so this works for large d
        # The real constraint is the lower bound: d >= 1/remaining
        
        # Let's try d from d_start up to some reasonable limit
        # For efficiency, cap the search
        max_d_to_try = min(10000000, max(d_start, int(remaining.denominator / remaining.numerator) + 100))
        
        # But this could still be huge. Let's be smarter.
        # Actually, for the "can complete" check:
        # remaining - 1/d <= (count-1)/(d+1)
        # Let R = remaining
        # R - 1/d <= (count-1)/(d+1)
        # This gives us an upper bound on d
        
        # Also: R - 1/d >= 0 → d >= 1/R
        # And: R - 1/d >= (count-1)/10000000 (usually not binding)
        
        # Let me just iterate with a cap
        for d in range(d_start, min(max_d_to_try + 1, 10000001)):
            if time.time() > deadline:
                return None
            if d < 5:
                continue
            if d in path:
                continue
            frac = Fraction(1, d)
            if frac > remaining:
                # d too small, 1/d > remaining, can't use
                continue
            new_remaining = remaining - frac
            if new_remaining < 0:
                continue
            # Feasibility check: can (count-1) terms from d+1 sum to new_remaining?
            if count - 1 == 0:
                if new_remaining == 0:
                    return list(path) + [d]
                continue
            # Max sum: (count-1)/(d+1)
            if new_remaining > Fraction(count - 1, d + 1):
                continue
            # Min sum: (count-1)/10000000
            # if new_remaining < Fraction(count-1, 10000000):
            #     continue  # Usually not binding
            
            path.append(d)
            result = dfs(new_remaining, d + 1, count - 1, path)
            path.pop()
            if result is not None:
                return result
        
        return None
    
    # Try n from 3 upward
    for n in range(3, 10):  # Try up to n=9
        if time.time() > deadline:
            break
        result = dfs(Fraction(1), 5, n, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == n and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: DP over divisors of 2520
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total in list(reached.keys()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = reached[total] + [d]
        if target in reached:
            sol = reached[target]
            # Verify
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and all(5 <= d <= 10000000 for d in sol) and len(set(sol)) == len(sol):
                return sol
    
    # Last resort fallback
    # Use a known valid representation
    # 1 = 1/5 + 1/6 + 1/7 + 1/420? No, that doesn't work.
    # Let's use: find any valid set
    # Simple: 1 = 1/6 + 1/7 + 1/14 + 1/210? 
    # 1/6+1/7 = 13/42, 1-13/42=29/42. 1/14=3/42. 29/42-3/42=26/42=13/21. 1/210? 13/21 = 130/210. No.
    
    # Use the DP result if it exists
    if target in reached:
        sol = reached[target]
        s = sum(Fraction(1, d) for d in sol)
        if s == 1:
            return sol
    
    # Construct a valid solution manually
    # 1 = 1/5 + 1/5 + ... no, distinct.
    # Let's use: 1 = 1/6 + 1/7 + 1/42 + 1/42? No, distinct.
    
    # Known: 1 = 1/5 + 1/6 + 1/7 + 1/420 doesn't work.
    # Let me compute: 1 - 1/5 - 1/6 - 1/7 = (210-42-35-30)/210 = 103/210. Not a unit fraction.
    
    # 1 - 1/5 - 1/6 - 1/8 = (120-24-20-15)/120 = 61/120. Not unit.
    # 1 - 1/5 - 1/6 - 1/9 = (90-18-15-10)/90 = 47/90. Not unit.
    # 1 - 1/5 - 1/6 - 1/10 = (30-6-5-3)/30 = 16/30 = 8/15. Not unit.
    # 1 - 1/5 - 1/6 - 1/11 = (330-66-55-30)/330 = 179/330. Not unit.
    
    # Let me just trust the DP
    if best_solution is None:
        raise RuntimeError("no valid representation found")
    
    raise RuntimeError("no valid representation found")
