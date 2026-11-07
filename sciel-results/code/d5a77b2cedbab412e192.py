"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    divisors = P['divisors']
    max_val = P['maximum']
    n = len(divisors)
    
    best_width = float('inf')
    best_result = None
    
    # Strategy: For a given window [L, L+W], find the smallest multiple of each divisor >= L.
    # We want to minimize W = max(multiples) - min(multiples).
    # 
    # Key insight: The optimal solution has a small span. We can search over possible
    # "base" values and check if we can fit all multiples in a tight window.
    #
    # More efficient approach: Enumerate possible assignments of multiples.
    # For each divisor d, the multiples in range [1, max_val] are d, 2d, 3d, ...
    # We need to pick one multiple per divisor such that they are all distinct and
    # the span is minimized.
    #
    # Since n=10 and the divisors are small, we can use a different strategy:
    # Try all possible "offsets" for each divisor relative to a base point.
    #
    # Better: Use the fact that the LCM of the divisors is large, but we only need
    # to find a local minimum. Let's use a randomized local search combined with
    # systematic scanning.
    #
    # Most efficient: For each possible "anchor" value A (the minimum multiple),
    # determine for each divisor d the smallest multiple >= A. Then compute the max.
    # The span is max - A. We want to minimize this.
    #
    # But A must be one of the multiples. So we can iterate over all possible values
    # that could be the minimum. However, there are too many.
    #
    # Alternative: Since the divisors are [6,10,14,15,21,22,26,33,35,39], let's
    # compute their LCM and use periodicity. But LCM is huge.
    #
    # Practical approach: The optimal span is likely small (< 100). For each possible
    # span W from 0 upward, check if there exists a window [L, L+W] such that each
    # divisor has a multiple in that window, and all multiples are distinct.
    #
    # For a fixed W, we can slide L. But we need to be smarter.
    #
    # Let's use: For each divisor d, the set of multiples in [L, L+W] is non-empty
    # iff there exists k such that L <= k*d <= L+W.
    # This is equivalent to: ceil(L/d) * d <= L+W, i.e., the smallest multiple >= L
    # is <= L+W.
    #
    # So for a given L and W, the condition for divisor d is:
    #   (ceil(L/d) * d) - L <= W
    # And we need all these multiples to be distinct.
    #
    # We can iterate W from 0 upward, and for each W, iterate L over a reasonable range.
    # But what range for L? The pattern repeats with period = LCM of divisors, which is huge.
    #
    # Better: Since we just need SOME valid assignment, and the divisors are small,
    # let's try a direct search over combinations.
    #
    # For each divisor d, the possible multiples are d, 2d, 3d, ... up to max_val.
    # We need to pick one per divisor, all distinct, minimizing max-min.
    #
    # Since n=10, and we can bound the search: the optimal span is probably < 200.
    # So each multiple is within a window of size ~200.
    #
    # Let's try: For each possible "base" B (the minimum value), for each divisor d,
    # the candidate multiple is the smallest multiple of d >= B. If this candidate
    # is <= B + W for some W, we have a candidate solution.
    #
    # We can iterate B over a range, say 1 to 100000, and for each B compute the
    # multiples and check distinctness and span.
    #
    # But 100000 iterations * 10 divisors = 1M operations, which is fast.
    #
    # Let's do that, but be smart: we only need to check B values that are multiples
    # of one of the divisors (since the minimum must be a multiple of some divisor).
    #
    # Actually, the minimum of the set is one of the multiples, so it's a multiple
    # of at least one divisor. But it could be any value.
    #
    # Simpler: Just iterate B from 1 to some limit, say 50000, and for each B,
    # compute the smallest multiple of each divisor >= B, check distinctness,
    # and compute span. Keep track of best.
    
    # First, do a quick systematic scan with a reasonable limit
    scan_limit = 50000
    
    B = 1
    while B <= scan_limit:
        if time.time() > deadline:
            break
        
        multiples = []
        valid = True
        for d in divisors:
            k = (B + d - 1) // d  # ceil(B/d)
            if k < 1:
                k = 1
            m = k * d
            if m > max_val:
                valid = False
                break
            multiples.append(m)
        
        if not valid:
            B += 1
            continue
        
        # Check distinctness
        if len(set(multiples)) != n:
            B += 1
            continue
        
        min_m = min(multiples)
        max_m = max(multiples)
        width = max_m - min_m
        
        if width < best_width:
            best_width = width
            best_result = list(multiples)
        
        B += 1
    
    # Now try a more targeted search: for small spans W, check if we can find
    # a valid assignment.
    # For W from 0 to 200, slide a window of size W and check.
    
    # Actually, the above scan already covers this implicitly. But let's add
    # a randomized local search to potentially find better solutions.
    
    # Randomized search: pick random starting points and do local optimization
    import random
    random.seed(42)
    
    # Try random base values
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        B = random.randint(1, max_val)
        multiples = []
        valid = True
        for d in divisors:
            k = (B + d - 1) // d
            if k < 1:
                k = 1
            m = k * d
            if m > max_val:
                valid = False
                break
            multiples.append(m)
        
        if not valid:
            continue
        
        if len(set(multiples)) != n:
            continue
        
        min_m = min(multiples)
        max_m = max(multiples)
        width = max_m - min_m
        
        if width < best_width:
            best_width = width
            best_result = list(multiples)
    
    if best_result is None:
        # Fallback: return the divisors themselves (valid but wide span)
        best_result = divisors[:]
    
    # Ensure types are correct: Python int
    best_result = [int(x) for x in best_result]
    
    return best_result
# EVOLVE-BLOCK-END
