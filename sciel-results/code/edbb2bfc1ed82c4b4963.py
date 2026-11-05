"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 87.0  # seconds, safety margin before 90s limit

def _check_x_is_mean_of_subset(x, S):
    """
    Check if x is the mean of any subset of at least 2 elements in S.
    Uses bitset DP on differences.
    """
    n = len(S)
    if n < 2:
        return False
    
    zs = [y - x for y in S]
    
    # Calculate max absolute sum for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False  # All equal to x, impossible with distinct elements
        
    offset = max_sum
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s-offset is achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
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
                
    return False

def _check_existing_violation(x, S):
    """
    Check if adding x to S creates any violation where an element a in S
    is the mean of a subset containing x and at least one other element from S.
    
    For each a in S, we need to check: is a the mean of {x} ∪ T where T ⊆ S\{a}, |T| >= 1?
    This means: |T|+1 elements sum to (|T|+1)*a
    So: sum(T) = |T|*a + a - x = (|T|+1)*a - x
    
    Equivalently: sum(T \ {a if a in T}) ... let me reformulate.
    
    Let U = S \ {a}. We need subset T ⊆ U with |T| >= 1 such that:
    x + sum(T) = (|T|+1) * a
    sum(T) = (|T|+1)*a - x
    
    Let w_t = t - a for t in U. Then sum(T) = sum_{t in T} w_t + |T|*a.
    So: sum_{t in T} w_t + |T|*a = (|T|+1)*a - x
    sum_{t in T} w_t = a - x
    
    So we need: does there exist a non-empty subset T ⊆ U such that sum_{t in T} (t-a) = a - x?
    
    This is a subset-sum problem on the values (t-a) for t in U, targeting (a-x).
    """
    n = len(S)
    if n < 2:
        return False
    
    target = None  # We'll compute per a
    
    for a in S:
        U = [t for t in S if t != a]
        if len(U) < 1:
            continue
        
        # We need non-empty subset T ⊆ U with sum_{t in T} (t - a) = a - x
        target_val = a - x
        if target_val == 0:
            # Need non-empty subset with sum 0, i.e., some t = a, but t != a in U
            # So check if any t in U equals a... but they don't since distinct and t != a
            # Actually sum of (t-a) = 0 means sum(t) = |T|*a, which is the original condition
            # Let's just run the DP
            pass
        
        ws = [t - a for t in U]
        max_abs = sum(abs(w) for w in ws)
        if max_abs == 0:
            continue
        
        offset = max_abs
        mask = (1 << (2 * max_abs + 1)) - 1
        
        # dp[k] = bitset for sums achievable with exactly k elements
        dp = [0] * (len(U) + 1)
        dp[0] = 1 << offset
        
        for w in ws:
            for k in range(len(U), 0, -1):
                if dp[k-1] == 0:
                    continue
                if w >= 0:
                    shifted = dp[k-1] << w
                else:
                    shifted = dp[k-1] >> (-w)
                dp[k] |= (shifted & mask)
        
        # Check if target_val is achievable with any k >= 1
        # target_val corresponds to bit (offset + target_val)
        if 0 <= offset + target_val <= 2 * max_abs:
            for k in range(1, len(U) + 1):
                if dp[k] & (1 << (offset + target_val)):
                    return True
    
    return False

def _quick_pair_check(x, S):
    """
    Quick check for 2-element subsets involving x.
    1. Is x the mean of a pair in S? (x = (a+b)/2 => 2x = a+b)
    2. Is any a in S the mean of pair (x, b) for b in S, b != a? (2a = x+b => b = 2a-x)
    """
    n = len(S)
    
    # 1. Is x mean of a pair in S?
    for i in range(n):
        for j in range(i+1, n):
            if S[i] + S[j] == 2 * x:
                return True
    
    # 2. Is any a in S mean of pair (x, b)?
    S_set = set(S)
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a:
            return True
    
    return False

def _full_check(C):
    """
    Full verification: check if any element in C is the mean of a subset of 
    at least 2 other elements.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        if _check_x_is_mean_of_subset(a, others):
            return True
    return False

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    orders = []
    
    # 1. Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # 2. Spread out orders
    spread_order = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order.append(hi)
        if lo != hi:
            spread_order.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    # Another spread: low, high, low+1, high-1, ...
    spread_order2 = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order2.append(lo)
        if lo != hi:
            spread_order2.append(hi)
        lo += 1
        hi -= 1
    orders.append(spread_order2)
    
    # 3. Random orders with different seeds
    rng = random.Random(42)
    num_random_orders = 300
    
    for i in range(num_random_orders):
        if time.time() - start_time > _DEADLINE - 3:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    # 4. Biased random orders: weight towards extremes
    for i in range(100):
        if time.time() - start_time > _DEADLINE - 3:
            break
        # Sample with bias towards 1 and 512
        biased = []
        for _ in range(512):
            r = rng.random()
            if r < 0.3:
                biased.append(rng.randint(1, 50))
            elif r < 0.6:
                biased.append(rng.randint(463, 512))
            else:
                biased.append(rng.randint(1, 512))
        # Remove duplicates while preserving order
        seen = set()
        unique_biased = []
        for v in biased:
            if v not in seen:
                seen.add(v)
                unique_biased.append(v)
        # Add any missing
        for v in range(1, 513):
            if v not in seen:
                unique_biased.append(v)
        orders.append(unique_biased)
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 2:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            if len(selected) >= 48:
                break
                
            # Quick pair check first (fast)
            if _quick_pair_check(x, selected):
                continue
            
            # Check if x is mean of any subset of >=2 in selected
            if _check_x_is_mean_of_subset(x, selected):
                continue
            
            # Check if any element in selected becomes mean of subset containing x
            if _check_existing_violation(x, selected):
                continue
            
            # Accept x
            selected.append(x)
            
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= 48:
                break
    
    # Final verification
    if best_set:
        if not _full_check(best_set):
            return best_set
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
