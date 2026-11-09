import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    L = 30030  # LCM of divisors
    
    # Precompute for each divisor d, which residues modulo L are multiples of d
    # residue_mask[i] is a set of residues r in [0, L) such that r % d_i == 0
    # But we need to be careful: 0 is a multiple, but we need positive integers
    # Actually, for a window [A, A+W], a value x in that window is a multiple of d
    # iff x % d == 0. Since x > 0, we just check x % d == 0.
    
    # Key insight: The pattern of multiples repeats every LCM = L.
    # For a window starting at A with width W, the candidates are A, A+1, ..., A+W.
    # A candidate at offset j (0-indexed) has value A+j.
    # (A+j) % d == 0 iff (A % d + j) % d == 0.
    
    # For each divisor d_i, and for each possible start residue A_mod = A % L,
    # we can precompute which offsets j in [0, W] are valid for divisor i.
    # But W varies, so let's precompute for each (i, A_mod) the sorted list of
    # offsets j in [0, L) such that (A_mod + j) % d_i == 0.
    
    # Actually, let's think differently. For a given window [A, A+W],
    # for divisor i, the valid values are multiples of d_i in [A, A+W].
    # The first multiple >= A is: first = ((A + d_i - 1) // d_i) * d_i
    # Then multiples are first, first + d_i, first + 2*d_i, ...
    
    # We'll use a different approach: iterate W from small to large.
    # For each W, iterate A from 0 to L-1 (since the pattern is periodic with period L).
    # For each (W, A), check if we can assign distinct multiples.
    
    # To speed up, use Hungarian algorithm or backtracking with pruning.
    # Since n=10 and W will be small (likely < 100), the number of candidates per divisor
    # is small (W/d_i + 1).
    
    # Let's precompute for each divisor d, the residues r in [0, d) where r % d == 0.
    # That's just r=0. So (A+j) % d == 0 iff (A % d + j) % d == 0.
    
    # For efficiency, let's just do the search with good pruning.
    
    # Precompute: for each divisor index i, and for each possible A_mod_d = A % d_i,
    # the offsets j such that (A_mod_d + j) % d_i == 0.
    # This is: j % d_i == (-A_mod_d) % d_i.
    # So the valid j are: j0, j0 + d_i, j0 + 2*d_i, ... where j0 = (-A_mod_d) % d_i.
    
    # Let's just iterate W and A, and for each, build the assignment problem.
    
    # Optimization: For a given W and A, for each divisor i, the number of candidates
    # is approximately W/d_i + 1. If W is small, this is very small.
    
    # We'll use backtracking with MRV (minimum remaining values) heuristic.
    
    def solve_for_window(A, W):
        """Try to find distinct multiples for all divisors in window [A, A+W].
        Returns list of 10 values or None."""
        
        # For each divisor i, find all multiples of d_i in [A, A+W] that are > 0
        candidates_per_div = []
        for i in range(n):
            d = divisors[i]
            # First multiple >= A
            if A == 0:
                first = d  # 0 is not > 0
            else:
                first = ((A + d - 1) // d) * d
            
            cands = []
            x = first
            while x <= A + W:
                if x > 0:
                    cands.append(x)
                x += d
            candidates_per_div.append(cands)
        
        # Quick check: if any has no candidates, fail
        if any(len(c) == 0 for c in candidates_per_div):
            return None
        
        # Backtracking with MRV
        # Order divisors by number of candidates (ascending)
        order = sorted(range(n), key=lambda i: len(candidates_per_div[i]))
        
        used = set()
        result = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            i = order[idx]
            for val in candidates_per_div[i]:
                if val not in used:
                    used.add(val)
                    result[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    result[i] = 0
            
            return False
        
        if backtrack(0):
            return result
        return None
    
    # Search for minimum W
    # W must be at least 9 (10 distinct integers need range >= 9)
    # But actually, the min difference between 10 distinct integers is at least 9.
    # So W >= 9.
    
    for W in range(9, L + 1):
        for A in range(0, L):
            res = solve_for_window(A, W)
            if res is not None:
                # Verify all are <= 1000000
                if max(res) <= 1000000:
                    return res
    
    # Fallback
    return divisors[:]
