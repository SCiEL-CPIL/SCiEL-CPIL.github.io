import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of the divisors
    LCM = 30030
    
    start_time = time.time()
    timeout = 85
    
    best_result = None
    best_range = float('inf')
    
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
        choices = [-1] * n
        
        def bt(idx, used):
            if idx == n:
                return True
            
            i = indices[idx]
            for off in candidates[i]:
                if off not in used:
                    used.add(off)
                    choices[i] = off
                    if bt(idx + 1, used):
                        return True
                    used.remove(off)
            
            return False
        
        used = set()
        if bt(0, used):
            return choices
        return None

    # Search for smallest W
    for W in range(9, 50):
        if time.time() - start_time > timeout:
            break
        
        # Precompute valid L ranges for this W
        # For each divisor d, L is valid if [L, L+W] contains a multiple of d.
        # This is equivalent to L being in [k*d - W, k*d] for some integer k.
        # We want the intersection of these valid sets for all divisors.
        # Since L is in [1, LCM], we can use a boolean array or difference array.
        
        # Difference array for counting valid intervals
        # Size LCM + 2
        diff = [0] * (LCM + 2)
        
        for d in divisors:
            # Find all k such that [k*d - W, k*d] intersects [1, LCM]
            # k*d >= 1 => k >= 1/d => k >= 1
            # k*d - W <= LCM => k <= (LCM + W) / d
            
            k_min = 1
            k_max = (LCM + W) // d
            
            for k in range(k_min, k_max + 1):
                start = k * d - W
                end = k * d
                
                # Clip to [1, LCM]
                if start < 1: start = 1
                if end > LCM: end = LCM
                
                if start <= end:
                    diff[start] += 1
                    diff[end + 1] -= 1
        
        # Compute prefix sum to find L where count == n
        valid_Ls = []
        count = 0
        for L in range(1, LCM + 1):
            count += diff[L]
            if count == n:
                valid_Ls.append(L)
        
        if not valid_Ls:
            continue
            
        # For each valid L, run backtracking
        found = False
        for L in valid_Ls:
            if time.time() - start_time > timeout:
                break
                
            # Get candidates for each divisor
            candidates = []
            for d in divisors:
                cand = get_candidates(d, L, W)
                # cand should not be empty because L is valid
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
            break
    
    # If no solution found within time, return the divisors themselves as fallback
    if best_result is None:
        return divisors[:]
    
    return best_result
