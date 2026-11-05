"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50  # 50s safety margin before 60s limit
    
    # Try to find minimal number of distinct denominators >= 5 summing to 1
    # Start with small counts and increase
    
    best = None
    
    # For each count k from 3 upward, try to find k distinct denominators >= 5
    # whose reciprocals sum to 1
    
    def search_k(k, min_den=5, start_idx=0, path=None):
        nonlocal best, deadline
        if path is None:
            path = []
        
        if time.time() > deadline:
            return None
        
        if len(path) == k:
            # Check if sum of reciprocals is 1
            total = Fraction(0, 1)
            for d in path:
                total += Fraction(1, d)
            if total == 1:
                return list(path)
            return None
        
        # Pruning: what's the max we can still add?
        remaining = k - len(path)
        # Each new term must be > last term in path
        last = path[-1] if path else min_den - 1
        
        # The remaining sum needed:
        # We need 1 - sum(path) to be representable as sum of 'remaining' distinct unit fractions > last
        
        # Lower bound on each remaining denominator: last+1, last+2, ...
        # So max possible sum of remaining = 1/(last+1) + 1/(last+2) + ... + 1/(last+remaining)
        # Min possible sum of remaining = 1/(10000000) + ... (very small)
        
        # For feasibility, we need the remaining sum to be achievable.
        # A necessary condition: the sum of the 'remaining' smallest possible terms 
        # (i.e., 1/(last+1) + ... + 1/(last+remaining)) must be >= the remaining sum needed.
        
        if len(path) > 0:
            current_sum = Fraction(0, 1)
            for d in path:
                current_sum += Fraction(1, d)
            needed = 1 - current_sum
        else:
            needed = 1
        
        # Max sum we can get from 'remaining' terms starting after 'last'
        max_sum = Fraction(0, 1)
        for i in range(remaining):
            max_sum += Fraction(1, last + 1 + i)
        
        if needed > max_sum:
            return None
        
        # Also, each term must be at least last+1, so min sum = sum of 1/(very large) ~ 0
        # But we need needed > 0
        
        if needed <= 0:
            return None
        
        # Try next denominator
        # To limit search space, we can bound the next denominator.
        # The next denominator d must satisfy 1/d >= needed/remaining (roughly)
        # Actually 1/d can be at most needed (if it's the only remaining term)
        # and at least needed/remaining (if all remaining are equal, but they're distinct)
        
        # Upper bound on d: 1/d >= needed/remaining is not quite right.
        # We need 1/d + (sum of remaining-1 terms) = needed
        # Each of the remaining-1 terms is < 1/d, so 1/d > needed/remaining
        # So d < remaining/needed... this gets complicated with Fractions.
        
        # Let's just try denominators starting from last+1 up to some reasonable bound.
        # For k=3, we can bound d reasonably.
        
        # Compute a practical upper bound for the next denominator
        # 1/d + 1/(d+1) + ... must cover 'needed'
        # So 1/d should be at least needed/remaining (conservative)
        # Actually, the largest term in the remaining must be at least needed/remaining
        # So 1/d >= needed/remaining → d <= remaining/needed
        
        # Convert needed to float for bound estimation
        needed_f = float(needed)
        if needed_f < 1e-15:
            return None
        
        # Upper bound for d
        if remaining > 0:
            d_max_est = int(remaining / needed_f) + 2
            if d_max_est < last + 1:
                d_max_est = last + 1
            # Cap to avoid huge searches
            if d_max_est > 200000:
                d_max_est = 200000
        else:
            d_max_est = last + 1
        
        for d in range(last + 1, min(d_max_est + 1, 10000001)):
            if time.time() > deadline:
                return None
            
            # Check if 1/d <= needed (necessary)
            if Fraction(1, d) > needed:
                continue
            
            result = search_k(k, min_den, 0, path + [d])
            if result is not None:
                return result
        
        return None
    
    # Try k = 3 first
    for k in range(3, 9):
        if time.time() > deadline:
            break
        result = search_k(k)
        if result is not None:
            best = result
            break
        if time.time() > deadline:
            break
    
    if best is None:
        # Fallback: use the DP approach from the seed
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                best = reached[target]
                break
    
    if best is None:
        raise RuntimeError("no representation found")
    
    return best

# EVOLVE-BLOCK-END
