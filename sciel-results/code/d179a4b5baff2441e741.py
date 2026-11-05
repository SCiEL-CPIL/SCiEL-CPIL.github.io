from fractions import Fraction

def construct():
    # Try to find a solution with minimal k
    # Based on harmonic series, k must be at least 8
    for k in range(8, 20):
        sol = search(k, Fraction(1), 4)
        if sol:
            return sol
    # Fallback: return a valid solution with more terms
    # We can generate one by taking 1/5..1/N and adjusting
    # But for now, raise error if not found
    raise RuntimeError("No solution found")

def search(k, target, last_d):
    if k == 0:
        return [] if target == 0 else None
    if target <= 0:
        return None
    
    # Pruning: max possible sum
    # Sum of 1/(last_d+1) + ... 
    # Approx k / (last_d + 1)
    if k / (last_d + 1) < target:
        return None
        
    min_d = max(last_d + 1, int(1 / target) + 1)
    max_d = int(k / target)
    
    for d in range(min_d, max_d + 1):
        nt = target - Fraction(1, d)
        if nt < 0:
            continue
        res = search(k - 1, nt, d)
        if res:
            return [d] + res
    return None
