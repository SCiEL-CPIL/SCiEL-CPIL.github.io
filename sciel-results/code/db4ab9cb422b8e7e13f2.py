"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

_divs = _get_divisors(5040)

def construct():
    deadline = time.time() + 55.0
    target = 4979
    
    # Try increasing k from 1 upward
    for k in range(1, 257):
        if time.time() > deadline:
            break
        # For small k, use combinations
        if k <= 6:
            from itertools import combinations
            for combo in combinations(_divs, k):
                if sum(combo) == target:
                    return list(combo)
        else:
            # For larger k, use a smarter approach
            # But first check if we can do it with fewer
            break
    
    # Fallback: greedy (should not reach here for this instance)
    remainder = target
    chosen = []
    for d in sorted(_divs, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen
# EVOLVE-BLOCK-END
