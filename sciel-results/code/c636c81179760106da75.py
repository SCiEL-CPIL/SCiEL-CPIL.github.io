from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 24
    # We process in decreasing order of value
    # S holds the selected elements
    S = []
    # sums holds all achievable subset sums of reciprocals of elements in S
    # We use Fraction for exactness
    sums = {Fraction(0, 1)}
    
    for u in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, u)
        
        # Check if target is already in sums
        if target in sums:
            # Conflict: 1/u is a sum of reciprocals of larger elements
            # Skip u
            continue
        
        # No conflict, add u
        S.append(u)
        
        # Update sums
        # New sums are old sums + 1/u
        # We only care about sums <= 1 (or <= max target which is 1)
        new_sums = set()
        inv_u = Fraction(1, u)
        for s in sums:
            ns = s + inv_u
            if ns <= 1:
                new_sums.add(ns)
        sums.update(new_sums)
        
        # Optional: Prune sums that are "too small" or "too large"?
        # Actually, we only need to check exact matches for future 1/v.
        # 1/v ranges from 1/24 to 1/1.
        # So we only need sums in [1/24, 1].
        # Sums < 1/24 cannot match any 1/v for v<=24.
        # So we can prune s < 1/24.
        min_target = Fraction(1, N)
        sums = {s for s in sums if s >= min_target or s == 0}
        
    # If S is empty, return at least one element (e.g., 24 is always safe)
    if not S:
        return [24]
        
    return S
