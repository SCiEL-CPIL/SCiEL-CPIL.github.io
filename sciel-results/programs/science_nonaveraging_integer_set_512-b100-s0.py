"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    Uses bitset DP.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    n = len(others)
    if n < 2:
        return False
    
    zs = [y - a for y in others]
    
    # Calculate max absolute sum for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        # All others are equal to a, which is impossible since elements are distinct
        # But if n >= 2, and all y == a, then they are not distinct from a?
        # The problem says distinct positive integers. So y != a.
        # So max_sum > 0.
        pass
        
    offset = max_sum
    # Mask to keep bitset within 2 * max_sum + 1 bits
    # We only care about sums in [-max_sum, max_sum]
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
    # We shift sums by offset, so sum 0 is at bit offset
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        # Iterate k downwards to avoid using the same element multiple times in one step
        # We are adding z to subsets of size k-1 to form subsets of size k
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            
            dp[k] |= (shifted & mask)
            
            # Early exit: if sum 0 is achievable with k >= 2 elements
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
                
    # Final check for any k >= 2
    for k in range(2, n + 1):
        if dp[k] & (1 << offset):
            return True
            
    return False

def _full_check(C):
    """
    Check if any element in C is the mean of a subset of at least 2 other elements.
    Returns True if a violation is found.
    """
    for i in range(len(C)):
        if _is_mean_of_subset(C, i):
            return True
    return False

def _fast_pair_check(C, new_idx):
    """
    Quick check for 2-element subsets (pairs).
    1. Is C[new_idx] the mean of a pair in C[:new_idx]?
    2. Is any C[i] (i < new_idx) the mean of a pair involving C[new_idx] and some C[j]?
    """
    x = C[new_idx]
    selected = C[:new_idx]
    n_sel = len(selected)
    
    # 1. Check if x is mean of a pair in selected
    # 2x = a + b
    # We can use a set for O(1) lookup, but n_sel is small.
    # Let's just do O(N^2) for simplicity, it's fast enough for N=48.
    for i in range(n_sel):
        for j in range(i+1, n_sel):
            if selected[i] + selected[j] == 2 * x:
                return True
                
    # 2. Check if any a in selected is mean of pair (x, y) where y in selected, y != a
    # 2a = x + y  =>  y = 2a - x
    # y must be in selected, y != a, y != x (x is not in selected yet, so y != x is automatic if y in selected)
    # Also y must be distinct from a.
    sel_set = set(selected)
    for a in selected:
        y = 2 * a - x
        if y in sel_set and y != a:
            return True
            
    return False

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate orders
    orders = []
    
    # 1. Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # 2. Random orders
    rng = random.Random(42)
    # Number of random orders depends on time budget
    # We will check time in the loop
    num_random_orders = 200
    
    for _ in range(num_random_orders):
        if time.time() - start_time > _DEADLINE - 5:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # 3. "Spread out" orders: pick from top and bottom
    # e.g., 512, 1, 511, 2, 510, 3, ...
    spread_order = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order.append(hi)
        if lo != hi:
            spread_order.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            # Fast pre-check
            # Create candidate list
            # To avoid list concatenation overhead, we can just check against selected
            # But _is_mean_of_subset needs the full list C.
            # Let's do the fast pair check first.
            
            # Fast pair check logic adapted for x and selected
            # 1. Is x mean of pair in selected?
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if selected[i] + selected[j] == 2 * x:
                        valid = False
                        break
                if not valid:
                    break
            
            # 2. Is any a in selected mean of pair (x, y)?
            if valid:
                sel_set = set(selected)
                for a in selected:
                    y = 2 * a - x
                    if y in sel_set and y != a:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Full check
            # Construct C = selected + [x]
            # This is O(N)
            C = selected + [x]
            
            # Run full check
            # This is the expensive part
            if _full_check(C):
                continue
                
            # Accept x
            selected.append(x)
            
            if len(selected) >= 48:
                break
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Final verification of best_set
    if best_set:
        if _full_check(best_set):
            # Fallback: return a known valid set
            # Powers of 2 are safe?
            # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
            # Mean of any subset of powers of 2 is not a power of 2?
            # Sum of distinct powers of 2 is unique.
            # If 2^k is mean of subset, sum = |S| * 2^k.
            # Sum of distinct powers of 2 has unique binary representation.
            # |S| * 2^k in binary is |S| followed by k zeros.
            # The sum of distinct powers of 2 < 2^(k+1) is < 2^(k+1).
            # It seems powers of 2 are a valid set.
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
