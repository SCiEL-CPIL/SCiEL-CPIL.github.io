"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # We want a nonempty subset of [1, 255] containing 255,
    # with gcd(a,b) > 1 for all distinct pairs, maximizing cardinality.
    
    # 255 = 3 * 5 * 17.
    # The set of all multiples of the smallest prime factor (3) in [1, 255]
    # forms a valid clique of size floor(255/3) = 85.
    # This is likely the maximum possible size for this problem.
    
    # Initialize with the set of multiples of 3.
    # This ensures we start with a large, valid clique.
    clique = set(i for i in range(1, 256) if i % 3 == 0)
    
    # Identify remaining candidates that share a factor with 255 but are not in the clique.
    # These are numbers divisible by 5 or 17 but not 3.
    remaining = [i for i in range(1, 256) if i not in clique and math.gcd(i, 255) > 1]
    
    # Attempt to expand the clique by adding compatible candidates.
    # A candidate c is compatible if gcd(c, x) > 1 for all x in clique.
    # Since 3 is in the clique, any c not divisible by 3 will have gcd(c, 3) = 1.
    # Thus, no candidate from 'remaining' (which are not divisible by 3) can be added.
    # However, we run the check to be robust and to follow the "greedy expansion" heuristic.
    
    while remaining:
        best_candidate = None
        # We want to find a candidate compatible with all current clique members.
        # Optimization: If 3 is in clique, any c not divisible by 3 is incompatible.
        # We can filter remaining to only those divisible by 3, but we already took all multiples of 3.
        # So remaining contains no multiples of 3.
        # Therefore, the loop will find no compatible candidates and break immediately.
        
        for c in remaining:
            # Check compatibility with all elements in clique
            if all(math.gcd(c, x) > 1 for x in clique):
                best_candidate = c
                break
        
        if best_candidate is None:
            break
        
        clique.add(best_candidate)
        remaining.remove(best_candidate)
        
    return list(clique)
# EVOLVE-BLOCK-END
