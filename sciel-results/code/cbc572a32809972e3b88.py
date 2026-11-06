# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    
    # We search for the minimum M such that target can be represented
    # as a sum of distinct unit fractions with denominators in [7, M].
    
    # parent[sum] = (prev_sum, denom)
    # We start with sum=0.
    
    parent = {Fraction(0): (None, None)}
    reachable = {Fraction(0)}
    
    # We iterate M from 7 upwards.
    # The maximum M we might need is unknown, but 10000 is a safe upper bound 
    # based on the problem constraints (denominators <= 10000000).
    # However, the optimal M is likely much smaller.
    
    for M in range(7, 10000):
        # Add 1/M to the available denominators
        # This generates new sums by adding 1/M to any existing reachable sum.
        
        new_sums = []
        for s in reachable:
            ns = s + Fraction(1, M)
            if ns <= target:
                if ns not in parent:
                    parent[ns] = (s, M)
                    new_sums.append(ns)
        
        if new_sums:
            reachable.update(new_sums)
            
        if target in parent:
            # Reconstruct the solution
            path = []
            curr = target
            while parent[curr][1] is not None:
                prev_sum, denom = parent[curr]
                path.append(denom)
                curr = prev_sum
            return path
        
    return []
# EVOLVE-BLOCK-END
