"""Optimized solver for packing distinct multiples of a fixed integer set."""
import math
import time
from itertools import product

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
LCM = 30030  # lcm of the divisors

def construct():
    """
    Find 10 distinct positive integers <= 1000000, each divisible by the
    corresponding divisor in DIVISORS, minimizing max - min.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    divs = DIVISORS
    n = len(divs)
    
    # Precompute residues for each divisor
    # For a given window [L, L+S], divisor d is feasible if there exists
    # a multiple of d in [L, L+S]
    # That is: there exists k such that L <= k*d <= L+S
    # Equivalent: ceil(L/d) * d <= L + S
    
    # Strategy: binary search on S (the span). For a given S, check if there
    # exists L such that for all i, there's a multiple of divs[i] in [L, L+S],
    # and all chosen multiples are distinct.
    
    # Actually, the distinctness constraint makes this harder. Let's think differently.
    
    # Approach: The optimal solution has all numbers within some interval of length S.
    # For each divisor d_i, the multiples in [L, L+S] are: ceil(L/d_i)*d_i, (ceil(L/d_i)+1)*d_i, ...
    # up to floor((L+S)/d_i)*d_i.
    
    # Key insight: If we fix L mod LCM, the pattern of available multiples repeats every LCM.
    # So we can search over L mod LCM (30030 possibilities) and for each, find the best S.
    
    # Better approach: For a fixed S, we need to find if there exists L such that:
    # 1. For each i, there is at least one multiple of divs[i] in [L, L+S]
    # 2. We can pick one multiple for each i such that all picks are distinct
    
    # Let's try a different angle: enumerate possible sets of residues.
    # For divisor d_i, in any interval of length S >= d_i - 1, there is at least one multiple.
    # The minimum S must be at least max(d_i) - 1 = 39 - 1 = 38? No, that's not right either.
    
    # Actually, the minimum possible span is bounded below by the fact that we need 10 distinct numbers.
    # The smallest span would be 9 if we could pack them consecutively, but divisibility constraints prevent this.
    
    # Let's use a direct search approach:
    # For each possible starting residue r in [0, LCM), check if we can find multiples.
    # For a given r and span S, L = r + k*LCM for some k >= 0.
    # The multiples of d_i in [L, L+S] depend on L mod d_i = (r + k*LCM) mod d_i = (r + k*(LCM mod d_i)) mod d_i.
    # Since LCM is divisible by d_i, LCM mod d_i = 0, so L mod d_i = r mod d_i.
    # So the pattern of available multiples depends only on r, not on k!
    
    # This means: for a fixed r, the set of "offsets" of multiples from L is fixed.
    # Specifically, for divisor d_i, the first multiple >= L is at offset:
    #   offset_i = (d_i - (r % d_i)) % d_i
    # and subsequent multiples are at offset_i + d_i, offset_i + 2*d_i, etc.
    # A multiple is in [L, L+S] iff offset_i + m*d_i <= S for some m >= 0.
    
    # So for fixed r, the available offsets for divisor i are:
    #   {offset_i, offset_i + d_i, offset_i + 2*d_i, ...} ∩ [0, S]
    
    # We need to pick one offset from each set such that all picked offsets are distinct.
    # This is a matching/assignment problem.
    
    # For a given r and S, we can check feasibility via backtracking or matching.
    # Then we binary search on S for each r, and take the minimum over r.
    
    # But 30030 values of r * binary search * matching might be slow.
    # Let's optimize: for a given r, the minimum S is determined by the "tightest" constraint.
    
    # Let's try: for each r in range(LCM), compute the minimum S needed.
    # For each divisor i, the available offsets are offset_i, offset_i+d_i, ...
    # We need to assign distinct offsets to all 10 divisors.
    # The minimum S is the minimum over all valid assignments of the max offset used.
    
    # This is still complex. Let's try a simpler heuristic first:
    # For each r, greedily assign the smallest available offset to each divisor,
    # ensuring distinctness.
    
    best_S = float('inf')
    best_assignment = None
    
    # For each residue r, try to find the minimal span
    for r in range(LCM):
        if time.time() > deadline - 5.0:
            break
        
        # Compute base offsets
        offsets = []
        for d in divs:
            base = (d - (r % d)) % d
            offsets.append(base)
        
        # Now we need to pick distinct offsets from {base_i, base_i + d_i, base_i + 2*d_i, ...}
        # to minimize the maximum.
        
        # Greedy: sort divisors by their base offset, assign smallest available
        # But this greedy may not be optimal. Let's use a smarter approach.
        
        # For a candidate S, check if we can assign distinct offsets <= S.
        # Binary search on S for this r.
        
        # Lower bound: max of the 10 smallest distinct values we can form
        # Upper bound: max(offsets) + max(divs) * 10 (safe upper bound)
        
        lo = 0
        hi = max(offsets) + max(divs) * 20
        
        def feasible(S_val, r_val, divs_list, offsets_list):
            # Check if we can assign distinct offsets <= S_val
            # For each divisor i, available offsets: offsets_list[i] + k * divs_list[i] for k >= 0, all <= S_val
            # We need to pick one from each such that all are distinct.
            
            # Use backtracking with pruning
            n_items = len(divs_list)
            # Sort by number of available options (ascending) for efficiency
            avail = []
            for i in range(n_items):
                opts = []
                v = offsets_list[i]
                while v <= S_val:
                    opts.append(v)
                    v += divs_list[i]
                avail.append(opts)
            
            # If any has no options, infeasible
            if any(len(a) == 0 for a in avail):
                return False
            
            # Backtracking
            avail_sorted = sorted(range(n_items), key=lambda i: len(avail[i]))
            used = set()
            
            def backtrack(idx):
                if idx == n_items:
                    return True
                i = avail_sorted[idx]
                for v in avail[i]:
                    if v not in used:
                        used.add(v)
                        if backtrack(idx + 1):
                            return True
                        used.remove(v)
                return False
            
            return backtrack(0)
        
        # Binary search for minimal S
        # But this is expensive. Let's try a different approach.
        
        # Actually, let's just try small S values directly for each r.
        # The answer is likely small (maybe < 100).
        
        # Try S from a reasonable lower bound upward
        # Lower bound: we need 10 distinct non-negative integers, so S >= 9
        # Also, S must be at least max(offsets) - min(offsets) potentially
        
        for S_test in range(9, 200):
            if time.time() > deadline - 2.0:
                break
            
            if feasible(S_test, r, divs, offsets):
                if S_test < best_S:
                    best_S = S_test
                    best_assignment = r
                break  # Found minimal S for this r, move to next r
    
    # Now we have best_S and best r. Let's construct the actual values.
    # We need to find L = r + k*LCM such that all multiples are <= MAX_VAL and positive.
    
    if best_S < float('inf') and best_assignment is not None:
        r = best_assignment
        S = best_S
        
        # Recompute offsets
        offsets = []
        for d in divs:
            base = (d - (r % d)) % d
            offsets.append(base)
        
        # Find an assignment of distinct offsets
        n_items = len(divs)
        avail = []
        for i in range(n_items):
            opts = []
            v = offsets[i]
            while v <= S:
                opts.append(v)
                v += divs[i]
            avail.append(opts)
        
        avail_sorted = sorted(range(n_items), key=lambda i: len(avail[i]))
        used = set()
        assignment = [None] * n_items
        
        def backtrack2(idx):
            if idx == n_items:
                return True
            i = avail_sorted[idx]
            for v in avail[i]:
                if v not in used:
                    used.add(v)
                    assignment[i] = v
                    if backtrack2(idx + 1):
                        return True
                    used.remove(v)
                    assignment[i] = None
            return False
        
        backtrack2(0)
        
        # Find k such that L = r + k*LCM gives valid multiples
        # L must be positive, and L + offset_i <= MAX_VAL for all i
        # Also L + offset_i > 0 for all i (automatically true if L > 0)
        
        # L = r + k*LCM
        # L + max(assignment) <= MAX_VAL
        # r + k*LCM + max(assignment) <= MAX_VAL
        # k <= (MAX_VAL - r - max(assignment)) / LCM
        
        max_offset = max(assignment)
        k_max = (MAX_VAL - r - max_offset) // LCM
        k_min = 0  # L = r >= 0, but we need L > 0 for positive multiples
        
        # L = r + k*LCM must be > 0
        # If r == 0, k >= 1; if r > 0, k >= 0
        
        if r == 0:
            k_start = 1
        else:
            k_start = 0
        
        if k_start <= k_max:
            k = k_start
            L = r + k * LCM
            result = [L + assignment[i] for i in range(n_items)]
            
            # Verify
            assert len(set(result)) == 10
            assert all(1 <= result[i] <= MAX_VAL for i in range(n_items))
            assert all(result[i] % divs[i] == 0 for i in range(n_items))
            
            return [int(x) for x in result]
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return [int(d) for d in divs]
# EVOLVE-BLOCK-END
