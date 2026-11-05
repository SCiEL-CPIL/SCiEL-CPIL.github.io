"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    # Try greedy Egyptian fraction first - fast and often gives minimal or near-minimal
    def greedy_egyptian(target, min_denom=5, max_terms=256):
        denoms = []
        last_d = min_denom - 1
        remaining = target
        while remaining > 0 and len(denoms) < max_terms:
            if time.time() > deadline:
                return None
            
            # Find smallest d > last_d such that 1/d <= remaining
            # 1/d <= remaining => d >= ceil(1/remaining)
            if remaining == 0:
                break
            d = max(last_d + 1, math.ceil(1 / remaining))
            
            # Verify 1/d <= remaining (handle floating point edge cases)
            while Fraction(1, d) > remaining:
                d += 1
            
            if d > 10000000:
                return None
            
            denoms.append(d)
            remaining -= Fraction(1, d)
            last_d = d
        
        if remaining != 0:
            return None
        if len(denoms) < 1 or len(denoms) > 256:
            return None
        if any(d < 5 or d > 10000000 for d in denoms):
            return None
        if len(set(denoms)) != len(denoms):
            return None
        s = sum(Fraction(1, d) for d in denoms)
        if s != 1:
            return None
        return denoms

    # Try greedy first
    result = greedy_egyptian(Fraction(1, 1), 5)
    if result is not None:
        return result

    # If greedy fails, try DFS for k=10 (known minimal)
    def max_sum_from(start, n):
        """Upper bound: sum of 1/(start) + 1/(start+1) + ... + 1/(start+n-1)"""
        if n <= 0:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for i in range(n):
            s += Fraction(1, start + i)
        return s

    def solve_k(k_val):
        def dfs(start, remaining, target, current):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return current if target == 0 else None
            if target <= 0:
                return None
            
            # Pruning: max possible sum with 'remaining' terms starting from 'start'
            max_possible = max_sum_from(start, remaining)
            if max_possible < target:
                return None
            
            # Bounds on d
            min_d = max(start, math.ceil(1 / target))
            max_d = int(remaining / target) if target > 0 else 10000000
            max_d = min(max_d, 10000000)
            
            if min_d > max_d:
                return None
            
            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return None
                
                new_target = target - Fraction(1, d)
                if new_target < 0:
                    break
                
                if remaining - 1 > 0:
                    max_next = max_sum_from(d + 1, remaining - 1)
                    if max_next < new_target:
                        break
                
                res = dfs(d + 1, remaining - 1, new_target, current + [d])
                if res is not None:
                    return res
            
            return None
        
        return dfs(5, k_val, Fraction(1, 1), [])

    # Try k=10 first (proven minimal)
    for k in range(10, 15):
        if time.time() > deadline - 5:
            break
        result = solve_k(k)
        if result and len(result) == k:
            if all(5 <= d <= 10000000 for d in result) and len(set(result)) == k:
                s = sum(Fraction(1, d) for d in result)
                if s == 1:
                    return result

    # Last resort: return empty (should not happen)
    return []
# EVOLVE-BLOCK-END
