"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    
    # Precompute: for each divisor, the list of possible "offsets" relative to a window start.
    # A number x is a multiple of d if x % d == 0.
    # If we fix a window [L, L+W], we need to pick one multiple of each divisor within [L, L+W].
    # Let x_i = L + offset_i, where 0 <= offset_i <= W.
    # Then (L + offset_i) % d_i == 0 => L % d_i == (-offset_i) % d_i.
    
    # We will search over window width W and start position L.
    # To make this efficient, we can fix the residues of L modulo each d_i.
    # There are prod(d_i) possible residue combinations, which is too large.
    
    # Alternative approach: 
    # 1. Sort divisors.
    # 2. Use a sliding window of multiples.
    #    For a given W, check if there exists an L such that for all i, there is a multiple of d_i in [L, L+W].
    #    This is equivalent to: the intersection of sets {L : exists k, L <= k*d_i <= L+W} for all i is non-empty.
    #    The set of valid L for divisor d_i is a union of intervals [k*d_i - W, k*d_i] for all k.
    
    # Even better: iterate over possible window starts L that are "aligned" with some multiple.
    # The optimal window will have its left edge at a multiple of some divisor, or its right edge at a multiple of some divisor.
    # Actually, the optimal L can be assumed to be of the form m*d_i - W for some multiple m*d_i, or m*d_i.
    
    # Let's try a different strategy:
    # Since the divisors are fixed and small, we can use a meet-in-the-middle or direct search over residues.
    
    # Direct search over L modulo lcm of some subset?
    # LCM of all divisors is huge.
    
    # Let's use a randomized + local search approach.
    # 1. Pick a random starting point.
    # 2. For each divisor, pick the closest multiple to the center of the window.
    # 3. Compute the span.
    # 4. Try to shift the window to reduce the span.
    
    # Even better: exhaustive search over small windows.
    # The minimum span is likely small. Let's try W from 1 upwards.
    # For a given W, we need to find if there exists L such that all divisors have a multiple in [L, L+W].
    # This is a constraint satisfaction problem.
    
    # We can model it as: for each divisor d_i, the valid L's are in the union of intervals [k*d_i - W, k*d_i].
    # We need the intersection of these unions to be non-empty.
    
    # Since W is small, each union is a collection of short intervals.
    # We can merge intervals for each divisor and check intersection.
    
    best_span = float('inf')
    best_assignment = None
    
    # Try small spans first
    for W in range(1, 1000):
        if time.time() > deadline:
            break
            
        # For each divisor, compute the intervals of valid L
        # Interval: L must satisfy L <= k*d_i <= L+W => k*d_i - W <= L <= k*d_i
        # So valid L for d_i is in [k*d_i - W, k*d_i] for some k.
        
        # We only need to consider k such that the interval overlaps with a reasonable range.
        # Let's assume L is in [1, 1000000].
        
        # For each divisor, generate the list of intervals.
        # Then find if there's a common point in all lists.
        
        # To do this efficiently, we can use a sweep line.
        # But first, let's limit the range of L.
        # The smallest possible L is 1, the largest is 1000000 - W.
        
        # Actually, we can just check a few candidate L's.
        # The optimal L will be close to a multiple of some divisor.
        
        # Candidate L's: for each divisor d_i and each multiple m*d_i, consider L = m*d_i - W and L = m*d_i.
        # This gives 10 * (1000000/max(d_i)) candidates, which is too many.
        
        # Better: for a given W, the valid L's form a periodic pattern with period LCM of divisors.
        # But LCM is huge.
        
        # Let's try a different approach: 
        # For each divisor, the set of valid L's is periodic with period d_i.
        # So the intersection is periodic with period LCM(d_i).
        # We can check one period.
        
        # LCM of [6,10,14,15,21,22,26,33,35,39] is very large.
        
        # Let's use a randomized approach with local optimization.
        if W < 50:  # Only do exhaustive-ish search for small W
            # For small W, we can check all L in a small range.
            # Actually, let's check L from 1 to max(d_i)*2.
            # No, that's not sufficient.
            
            # Let's check L that are multiples of 1, 2, ..., up to some limit.
            # Actually, for small W, the number of valid L's is small.
            
            # Let's generate candidate L's:
            # For each divisor d_i, the valid L's are in [k*d_i - W, k*d_i].
            # For small W, these are short intervals.
            # The intersection of all these interval unions is non-empty iff there's a point in all of them.
            
            # We can use a sweep line over the candidate intervals.
            # But for small W, we can just check all integer L in a small range.
            
            # What range? The L's must be such that for all i, there's a multiple of d_i in [L, L+W].
            # The smallest such L is 1 (if 1 <= d_i <= 1+W for all i, which is not true).
            # Actually, L must be at least 1 and at most 1000000 - W.
            
            # Let's check L from 1 to 1000.
            found = False
            for L in range(1, 1001):
                assignment = []
                valid = True
                for d in divisors:
                    # Find multiple of d in [L, L+W]
                    # Smallest multiple >= L: ceil(L/d)*d
                    k = (L + d - 1) // d
                    m = k * d
                    if m > L + W:
                        valid = False
                        break
                    assignment.append(m)
                if valid and len(set(assignment)) == n:
                    # Check distinctness
                    if len(set(assignment)) == n:
                        span = max(assignment) - min(assignment)
                        if span < best_span:
                            best_span = span
                            best_assignment = assignment[:]
                        found = True
                        break  # First valid L for this W is enough
            if found:
                # Since we're iterating W from 1, the first W with a valid assignment is optimal
                # But we need to check if there's a smaller span with a larger W? No, span <= W.
                # Actually, span = max - min, and all values are in [L, L+W], so span <= W.
                # So the first W with a valid assignment gives span <= W.
                # But maybe a larger W gives a smaller span? No, span is at most W, so smaller W is better.
                # Wait, span is max - min, which is <= W. So minimizing W minimizes the upper bound on span.
                # But the actual span might be smaller than W.
                # So we should continue searching for smaller spans even with larger W?
                # No, if we find a valid assignment with span S, then any W < S cannot have a valid assignment.
                # So the optimal span is the minimum S such that there exists W >= S with a valid assignment.
                # But since span <= W, we can just search W from 1 upwards and record the minimum span.
                pass
        
        # For larger W, use a different approach
        if W >= 50 and W <= 200:
            # Use a smarter search
            # Check L values that are close to multiples of the divisors
            candidate_Ls = set()
            for d in divisors:
                for k in range(1, 100):  # Check first 100 multiples
                    m = k * d
                    candidate_Ls.add(max(1, m - W))
                    candidate_Ls.add(m)
            
            for L in candidate_Ls:
                if L < 1 or L > 1000000 - W:
                    continue
                assignment = []
                valid = True
                for d in divisors:
                    k = (L + d - 1) // d
                    m = k * d
                    if m > L + W:
                        valid = False
                        break
                    assignment.append(m)
                if valid and len(set(assignment)) == n:
                    span = max(assignment) - min(assignment)
                    if span < best_span:
                        best_span = span
                        best_assignment = assignment[:]
    
    if best_assignment is not None:
        # Ensure types are correct
        return [int(x) for x in best_assignment]
    else:
        # Fallback: use the divisors themselves
        return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
