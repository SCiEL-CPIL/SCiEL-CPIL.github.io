"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    # Greedy Egyptian fraction with minimum denominator 5
    # We want to minimize the number of terms
    # Try greedy approach: always take the largest unit fraction <= remaining
    # with denominator >= max(5, previous_denom + 1)
    
    remaining = Fraction(1, 1)
    used = set()
    result = []
    min_denom = 5
    
    while remaining > 0:
        # Find the smallest denominator d >= min_denom such that 1/d <= remaining
        # i.e., d >= 1/remaining
        d = max(min_denom, -(-1 // remaining.numerator) * remaining.denominator) if remaining.numerator > 0 else min_denom
        # More carefully: we need d such that 1/d <= remaining, i.e., d >= 1/remaining
        # 1/remaining = remaining.denominator / remaining.numerator
        # So d >= ceil(remaining.denominator / remaining.numerator)
        import math
        min_d = math.ceil(remaining.denominator / remaining.numerator)
        d = max(min_denom, min_d)
        
        # Check if 1/d <= remaining
        while Fraction(1, d) > remaining:
            d += 1
        
        # Use this denominator
        result.append(d)
        used.add(d)
        remaining -= Fraction(1, d)
        min_denom = d + 1  # next must be strictly larger
        
        if len(result) > 256:
            raise RuntimeError("too many terms")
    
    # Verify all denominators in [5, 10000000]
    for d in result:
        if d < 5 or d > 10000000:
            raise RuntimeError(f"denominator {d} out of range")
    
    # Verify sum is exactly 1
    total = sum(Fraction(1, d) for d in result)
    if total != 1:
        raise RuntimeError(f"sum is {total}, not 1")
    
    return result

# EVOLVE-BLOCK-END
