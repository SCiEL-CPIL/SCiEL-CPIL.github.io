import itertools
import time

def construct():
    # Attempt to construct a larger set by adding 3 to the previous candidate
    # Previous: {7, 9, 10, ..., 24} (size 17)
    # New candidate: {3, 7, 9, 10, ..., 24} (size 18)
    
    candidate = [3, 7] + list(range(9, 25))
    
    # Verification
    recips = {i: 1.0/i for i in candidate}
    valid = True
    for x in candidate:
        target = recips[x]
        others = [y for y in candidate if y != x]
        # Check if target is a subset sum of reciprocals of others
        # Using a simple DP or backtracking for subset sum might be slow if done naively for all x
        # But n=18, subsets are 2^17 ~ 131k. 18 * 131k ~ 2.3M operations. Fast enough.
        
        # However, floating point subset sum is tricky.
        # We can use exact rational arithmetic with fractions.Fraction for safety.
        
        from fractions import Fraction
        target_frac = Fraction(1, x)
        other_fracs = [Fraction(1, y) for y in others]
        
        # Generate all subset sums of other_fracs
        # To optimize, we can prune sums that exceed target_frac
        # Since all fractions are positive, we can stop if sum > target
        
        sums = set()
        sums.add(Fraction(0, 1))
        
        for frac in other_fracs:
            new_sums = set()
            for s in sums:
                new_s = s + frac
                if new_s <= target_frac:
                    new_sums.add(new_s)
            sums.update(new_sums)
            
        if target_frac in sums:
            valid = False
            break
            
    if valid:
        return candidate
    else:
        # Fallback to the previous known good set
        return [7] + list(range(9, 25))
