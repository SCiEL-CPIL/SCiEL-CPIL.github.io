"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _greedy_egyptian(max_terms=256, deadline=85.0):
    """Greedy Egyptian fraction: pick smallest unused d>=5 with 1/d <= remaining."""
    start = time.time()
    remaining = Fraction(1, 1)
    used = set()
    result = []
    
    for step in range(max_terms):
        if time.time() > deadline:
            break
        if remaining == 0:
            break
        # Find smallest unused d >= 5 such that 1/d <= remaining
        d = 5
        while d <= 10000000:
            if d not in used:
                if Fraction(1, d) <= remaining:
                    # Check if remaining - 1/d can still be completed
                    new_rem = remaining - Fraction(1, d)
                    if new_rem == 0:
                        used.add(d)
                        result.append(d)
                        return result
                    # Sufficient condition: new_rem <= (max_terms - len(result) - 1) / 5
                    terms_left = max_terms - len(result) - 1
                    if terms_left <= 0:
                        break
                    # Also need new_rem > 0
                    if new_rem > 0 and new_rem <= Fraction(terms_left, 5):
                        used.add(d)
                        result.append(d)
                        remaining = new_rem
                        break
            d += 1
        else:
            # No valid d found in range, try to finish with what we have
            # This shouldn't happen if our conditions are right
            break
    
    # Verify result
    if not result:
        return None
    s = sum(Fraction(1, d) for d in result)
    if s == 1 and len(result) <= 256 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
        return result
    return None

def _dp_fallback():
    """Original DP approach as fallback."""
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")

def construct():
    # Try greedy first
    result = _greedy_egyptian()
    if result is not None:
        return result
    # Fall back to DP
    return _dp_fallback()
# EVOLVE-BLOCK-END
