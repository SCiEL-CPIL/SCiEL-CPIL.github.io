"""Few-term Egyptian fractions with denominators greater than four."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _validate(denoms):
    if not denoms or len(denoms) > 256:
        return False
    if any(d < 5 or d > 10000000 for d in denoms):
        return False
    if len(set(denoms)) != len(denoms):
        return False
    total = sum(Fraction(1, d) for d in denoms)
    return total == 1

def _search_k_terms(k, deadline):
    """Search for k distinct denominators >= 5 summing to 1."""
    best = None
    
    def dfs(remaining, min_denom, current, depth):
        nonlocal best
        if time.time() > deadline:
            return
        if depth == k:
            if remaining == 0:
                best = list(current)
            return
        
        if remaining <= 0:
            return
        
        # Pruning: max possible sum with remaining terms
        slots = k - depth
        max_possible = sum(Fraction(1, min_denom + i) for i in range(slots))
        if remaining > max_possible:
            return
        
        # Min possible: 1/min_denom for first term
        d_start = max(min_denom, 5)
        # 1/d < remaining => d > 1/remaining
        # Use float for speed in bounds
        d_min_float = 1.0 / float(remaining)
        d_start = max(d_start, int(math.ceil(d_min_float)))
        
        # Upper bound: we need (slots-1) more terms each >= 1/(d+1)
        # So remaining - 1/d <= (slots-1) * 1/(d+1) approximately
        # Simpler: 1/d must be < remaining, and remaining - 1/d must be > 0
        # Also remaining - 1/d must be <= sum of 1/(d+1) + 1/(d+2) + ...
        
        # Try denominators
        # Limit search range for efficiency
        max_d = 200000  # reasonable upper bound for small k
        
        for d in range(d_start, max_d + 1):
            if time.time() > deadline:
                return
            if best is not None and len(current) + (k - depth) >= len(best):
                # Can't improve, but we're searching for exactly k terms so this check is moot
                pass
            
            term = Fraction(1, d)
            if term >= remaining:
                break
            
            new_rem = remaining - term
            if new_rem <= 0:
                continue
            
            # Check feasibility: can we make new_rem with (k-depth-1) terms from d+1?
            if k - depth - 1 > 0:
                max_rem = sum(Fraction(1, d + 1 + i) for i in range(k - depth - 1))
                if new_rem > max_rem:
                    continue
                # Min rem check: each term is positive, so new_rem > 0 (already checked)
            
            current.append(d)
            dfs(new_rem, d + 1, current, depth + 1)
            current.pop()
            if best is not None:
                return
    
    dfs(Fraction(1, 1), 5, [], 0)
    return best

def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 60s limit
    
    # Try to find shorter solutions first
    for k in range(8, 11):  # try 8, 9, 10 terms
        if time.time() > deadline:
            break
        result = _search_k_terms(k, deadline)
        if result is not None:
            if _validate(result):
                return result
    
    # Fallback: hardcoded 10-term solution
    denoms = [5, 6, 7, 8, 9, 10, 12, 15, 230, 57960]
    if _validate(denoms):
        return denoms
    
    return []

# EVOLVE-BLOCK-END
