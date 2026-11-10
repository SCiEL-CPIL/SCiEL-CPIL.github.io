import time
import math
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of the divisors
    LCM = 30030
    
    start_time = time.time()
    timeout = 85
    
    best_result = None
    best_range = float('inf')
    
    # Precompute for each divisor d and each possible L (mod LCM),
    # the list of offsets (relative to L) where multiples of d fall in [L, L+W]
    # We'll compute this on-the-fly but cache it.
    
    def get_candidates(d, L, W):
        """Get list of offsets (x - L) where x is a multiple of d in [L, L+W]"""
        k = (L + d - 1) // d
        first = k * d
        if first > L + W:
            return []
        cand = []
        x = first
        while x <= L + W:
            cand.append(x - L)
            x += d
        return cand
    
    def backtrack(candidates, indices, n):
        """
        candidates: list of lists of offsets for each divisor
        indices: order in which to assign (sorted by candidate count)
        Returns list of chosen offsets or None
        """
        choices = [None] * n
        used = set()
        
        def bt(idx):
            if idx == n:
                return True
            
            i = indices[idx]
            for off in candidates[i]:
                if off not in used:
                    used.add(off)
                    choices[i] = off
                    if bt(idx + 1):
                        return True
                    used.remove(off)
            
            return False
        
        if bt(0):
            return choices
        return None
    
    # Search for smallest W
    for W in range(9, 40):
        if time.time() - start_time > timeout:
            break
        
        # For each L in [1, LCM], check if a valid assignment exists
        found = False
        for L in range(1, LCM + 1):
            if time.time() - start_time > timeout:
                break
            
            # Quick feasibility check
            feasible = True
            for d in divisors:
                if (L + d - 1) // d * d > L + W:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Get candidates for each divisor
            candidates = []
            for d in divisors:
                cand = get_candidates(d, L, W)
                if not cand:
                    candidates = None
                    break
                candidates.append(cand)
            
            if candidates is None:
                continue
            
            # Sort indices by number of candidates (ascending) for better pruning
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            # Backtracking
            result = backtrack(candidates, indices, n)
            if result is not None:
                # Verify the solution
                values = [L + result[i] for i in range(n)]
                if len(set(values)) == n and all(1 <= v <= 1000000 for v in values):
                    if all(values[i] % divisors[i] == 0 for i in range(n)):
                        current_range = max(values) - min(values)
                        if current_range < best_range:
                            best_range = current_range
                            best_result = values
                        found = True
                        break
        
        if found:
            # If we found a solution with range W, that's optimal for this W
            # But we should continue to see if a smaller W exists (though we're iterating up)
            # Since we're iterating W from 9 upward, the first W with a solution is optimal
            break
    
    # If no solution found within time, return the divisors themselves as fallback
    if best_result is None:
        return divisors[:]
    
    return best_result
