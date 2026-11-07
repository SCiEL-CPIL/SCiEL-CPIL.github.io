"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import ceil

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    deadline = time.time() + 50  # safety margin before 90s limit
    
    best_result = None
    best_range = float('inf')
    
    # Try increasing window sizes
    for W in range(1, 200):  # Try windows up to size 199
        if time.time() > deadline:
            break
        
        # For each window size, try various starting positions
        # The starting position L should be such that multiples exist
        # We can iterate L over a reasonable range
        # For a given W, L can range widely, but we only need to check
        # L values that are "aligned" with the divisors' multiples
        
        # Key insight: for a window [L, L+W], the set of multiples for divisor d
        # depends on L mod d. So we only need to check L values that are distinct
        # modulo the LCM of divisors, but that's too large.
        
        # Alternative: for each divisor d, the multiples in [L, L+W] are determined
        # by which multiples of d fall in that range. As L increases by 1, the set
        # changes. But we can sample L values.
        
        # Better approach: for a fixed W, try L values from 0 to some limit,
        # but step by 1. However, this is too slow for large ranges.
        
        # Insight: the pattern of available multiples repeats with period = LCM of divisors,
        # but LCM is huge. Instead, note that for small W, each divisor has few candidates.
        
        # Let's try L in a range. Since we want small windows, L can be anything.
        # But the problem is symmetric in a sense - we can shift all values.
        # Actually no, because multiples are anchored to 0.
        
        # Practical approach: try L from 0 to 100000, step 1, for small W.
        # But that's 100k * 200 = 20M iterations, too slow.
        
        # Better: for each W, try L values that are multiples of small numbers or
        # just sample. Actually, let's try a different approach.
        
        # For a given W, the condition for divisor d to have at least one multiple
        # in [L, L+W] is that there exists k such that L <= k*d <= L+W.
        # This is always true for W >= d-1. For W < d-1, it's not always true.
        
        # Let's just try L from 0 to 1000000, but with early termination and
        # efficient checking.
        
        # Actually, let's be smarter. For a given W, we can determine for each
        # divisor d the "gaps" where no multiple exists. The window [L, L+W]
        # must contain at least one multiple of each d.
        
        # Let's try: for each W, iterate L from 0 to 50000 (or less), and for each L,
        # check if we can assign distinct multiples.
        
        # To speed up, precompute for each divisor d and each L, the list of
        # multiples in [L, L+W]. But that's expensive.
        
        # Alternative: for each L, compute the candidate multiples for each divisor.
        # If any divisor has no candidates, skip. Otherwise, do backtracking.
        
        # Let's limit L search to 0..20000 for now, and increase if needed.
        
        for L in range(0, 20001):
            if time.time() > deadline:
                break
            
            # Get candidates for each divisor
            candidates = []
            valid = True
            for d in divisors:
                # Find multiples of d in [L, L+W]
                k_min = ceil(L / d)
                k_max = (L + W) // d
                mults = [k * d for k in range(k_min, k_max + 1)]
                if not mults:
                    valid = False
                    break
                candidates.append(mults)
            
            if not valid:
                continue
            
            # Check if all candidates are within max_val
            # (they should be since L+W <= 20000+199 < 1000000)
            
            # Backtracking: assign distinct multiples to each divisor
            # Sort by number of candidates (most constrained first)
            order = sorted(range(len(divisors)), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * len(divisors)
            
            def backtrack(idx):
                if idx == len(divisors):
                    return True
                d_idx = order[idx]
                for m in candidates[d_idx]:
                    if m not in used:
                        used.add(m)
                        assignment[d_idx] = m
                        if backtrack(idx + 1):
                            return True
                        used.remove(m)
                        assignment[d_idx] = None
                return False
            
            if backtrack(0):
                # We found a valid assignment
                current_range = W  # since max - min <= W, and we're in [L, L+W]
                # Actually, the actual range might be less than W
                actual_vals = [assignment[i] for i in range(len(divisors))]
                actual_range = max(actual_vals) - min(actual_vals)
                
                if actual_range < best_range:
                    best_range = actual_range
                    best_result = [int(assignment[i]) for i in range(len(divisors))]
                    
                    if best_range == 1:  # Can't do better than 1 (distinct integers)
                        return best_result
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback: return the divisors themselves (which are distinct and valid)
        return [int(d) for d in divisors]
    
    return best_result

# EVOLVE-BLOCK-END
