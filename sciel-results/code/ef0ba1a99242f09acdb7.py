"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    # Use DP to find the minimal number of distinct denominators summing to 1.
    # We use 2520 as the common denominator target.
    target = 2520
    # reached[sum] = (min_terms, list_of_denominators)
    reached = {0: (0, [])}
    
    # Iterate through possible denominators (divisors of 2520, >= 5)
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        # Iterate over current states
        for total, (terms, values) in list(reached.items()):
            new_total = total + w
            if new_total > target:
                continue
            new_terms = terms + 1
            # Update if we found a path with fewer terms
            if new_total not in reached or new_terms < reached[new_total][0]:
                reached[new_total] = (new_terms, values + [d])
    
    if target in reached:
        result = reached[target][1]
        # Validate
        s = Fraction(0, 1)
        for d in result:
            s += Fraction(1, d)
        if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
            return result
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
