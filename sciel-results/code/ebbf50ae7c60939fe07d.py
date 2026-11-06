"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct."""
    k = len(subset)
    if k == 0:
        return True
    # Use fractions to avoid floating point issues
    from fractions import Fraction
    reciprocals = [Fraction(1, x) for x in subset]
    seen = set()
    # 2^k subset sums
    for mask in range(1 << k):
        s = Fraction(0, 1)
        for i in range(k):
            if mask & (1 << i):
                s += reciprocals[i]
        if s in seen:
            return False
        seen.add(s)
    return True

def construct():
    # Search for maximum cardinality dissociated subset of [1..22]
    best = None
    best_size = 0
    # We know from the seed that size 9 is achievable.
    # Try to find larger sizes. Max possible is 22 but realistically limited.
    # For k=10, 2^10=1024 sums, manageable.
    # For k=11, 2^11=2048 sums, manageable.
    # For k=12, 2^12=4096 sums, manageable.
    # For k=13, 2^13=8192 sums, manageable.
    # Let's try from high k down to find the maximum.
    
    from fractions import Fraction
    
    def check(subset):
        k = len(subset)
        reciprocals = [Fraction(1, x) for x in subset]
        seen = set()
        for mask in range(1 << k):
            s = Fraction(0, 1)
            for i in range(k):
                if mask & (1 << i):
                    s += reciprocals[i]
            if s in seen:
                return False
            seen.add(s)
        return True
    
    # Try sizes from 14 down to 9
    for k in range(14, 8, -1):
        found = False
        for subset in combinations(range(1, 23), k):
            if check(subset):
                best = list(subset)
                best_size = k
                found = True
                break
        if found:
            break
    
    if best is None:
        # Fallback to the seed solution
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best
# EVOLVE-BLOCK-END
