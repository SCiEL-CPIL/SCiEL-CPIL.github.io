"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Strategy: Build a set where min sum of k elements > max sum of k-1 elements
    # for all k. This guarantees no cross-cardinality sum collisions.
    # 
    # For sorted set a[0] < a[1] < ... < a[n-1]:
    # Condition: sum(a[0:k]) > sum(a[n-k+1:n]) for k=1..n
    # (k smallest vs k-1 largest)
    #
    # We'll use a two-pointer greedy: start with small and large values,
    # and ensure the gap condition holds.
    
    # Key insight: if we use powers of 2 or near-powers, we get strong separation
    # but waste range. Instead, use a geometric-like progression.
    
    # Let's try: build set greedily from high end downward, ensuring
    # that adding a new smallest element still maintains the condition.
    
    # Alternative: use the fact that if a[i+1] >= 2*a[i] + 1, then all subset sums
    # are unique (superincreasing), which implies our condition. But this limits size.
    # 1, 2, 4, 8, 16, 32, 64, 128, 256 -> only 9 elements in 1..256.
    
    # Better: we don't need all subset sums unique, just cross-cardinality separation.
    # We can have repeated sums within same cardinality.
    
    # Let's use a different approach: 
    # If we pick numbers such that the sum of the k smallest > sum of k-1 largest,
    # we can have a denser set.
    
    # Try: arithmetic progression with large step, or mixed.
    
    # Let me compute: for a set of size n from values in 1..256,
    # what's the maximum n we can achieve?
    
    # For an AP with first term a, last term b, n terms, step d:
    # b = a + (n-1)*d
    # min sum of k: k*a + d*k*(k-1)/2
    # max sum of k-1: (k-1)*b - d*(k-1)*(k-2)/2
    # Need: k*a + d*k*(k-1)/2 > (k-1)*b - d*(k-1)*(k-2)/2
    
    # For k=1: a > 0 (always true for a>=1)
    # For k=n: sum(all) > sum of n-1 largest = sum(all) - a[0]
    #   => a[0] > 0, always true
    # The tightest constraint is usually in the middle.
    
    # Let me just try various constructions and validate.
    
    def validate(s):
        """Check if set s satisfies the cross-cardinality sum separation."""
        s = sorted(s)
        n = len(s)
        # For each cardinality k from 1 to n, compute min and max sum
        # min sum of k elements: sum of k smallest
        # max sum of k elements: sum of k largest
        # Need: max_sum(k-1) < min_sum(k) for all k=1..n
        # (with max_sum(0) = 0, min_sum(0) = 0)
        
        # Precompute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + s[i]
        
        total = prefix[n]
        
        for k in range(1, n + 1):
            min_sum_k = prefix[k]  # k smallest
            max_sum_k_minus_1 = total - prefix[n - (k-1)]  # k-1 largest
            if k == 1:
                max_sum_k_minus_1 = 0  # empty subset
            if max_sum_k_minus_1 >= min_sum_k:
                return False
        return True
    
    # Try to build a large set. Start with a candidate and validate.
    
    # Approach: use a set that's dense in the middle but has enough spread.
    # Let's try: {a, a+d, a+2d, ..., a+(n-1)d} where we maximize n.
    
    best_set = list(range(1, 9))  # fallback: 1..8 works? Let's check.
    
    # Actually, let me try a smarter construction.
    # Use numbers that grow roughly geometrically but with some flexibility.
    
    # Try: start with high numbers and work down, ensuring separation.
    # Or: use a known good pattern.
    
    # Let me try: {1, 3, 7, 15, 31, 63, 127, 255} -> 8 elements (2^k - 1)
    # Check: min sum of k: sum(2^i - 1 for i=0..k-1) = 2^k - k
    # max sum of k-1: sum(2^i - 1 for i=n-k+1..n-1) where n=8
    # This is superincreasing-like, should work. But only 8 elements.
    
    # Can we do better? Let's try to pack more.
    
    # What if we use: {1, 2, 4, 7, 12, 21, 38, 70, 131, 250}?
    # Each is roughly 2x previous + 1. Let's validate.
    
    candidates = []
    
    # Try APs with various parameters
    for n in range(60, 8, -1):
        # Try to fit n elements in 1..256 with AP
        # a + (n-1)*d <= 256, a >= 1, d >= 1
        # For given n, max d = (256-1)//(n-1)
        max_d = (256 - 1) // (n - 1) if n > 1 else 255
        for d in range(max_d, 0, -1):
            a = 1
            # Check if AP starting at a with step d, n terms fits
            if a + (n-1)*d > 256:
                continue
            s = [a + i*d for i in range(n)]
            if validate(s):
                candidates.append(s)
                break
        if candidates:
            break
        # Try starting at different a
        for a in range(1, 10):
            max_d = (256 - a) // (n - 1) if n > 1 else 255
            for d in range(max_d, 0, -1):
                s = [a + i*d for i in range(n)]
                if s[-1] > 256:
                    continue
                if validate(s):
                    candidates.append(s)
                    break
            if candidates:
                break
        if candidates:
            break
    
    if candidates:
        # Return the largest one
        best = max(candidates, key=len)
        # Ensure Python ints
        return [int(x) for x in best]
    
    # If no AP worked, try a greedy construction
    # Start with empty set, add elements one by one (from large to small)
    # ensuring the condition still holds.
    
    s = []
    # Try adding elements from 256 down to 1
    for val in range(256, 0, -1):
        s.append(val)
        if not validate(s):
            s.pop()
        if len(s) >= 20:  # good enough if we can't do better
            break
    
    # Also try from small to large
    s2 = []
    for val in range(1, 257):
        s2.append(val)
        if not validate(s2):
            s2.pop()
        if len(s2) >= 20:
            break
    
    # Pick the better one
    if len(s2) > len(s):
        s = s2
    
    # Final validation
    if not validate(s):
        # Fallback to a known good set
        s = list(range(1, 9))
    
    return [int(x) for x in s]

# EVOLVE-BLOCK-END
