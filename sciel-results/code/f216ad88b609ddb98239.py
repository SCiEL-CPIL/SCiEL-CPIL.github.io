"""Program for large reciprocal-subsum-free sets using integer Diophantine subset-sum."""
import time
from math import gcd
from functools import reduce

_START_TIME = time.time()
_DEADLINE = 88.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_subset_sum_integer(target_int, weights):
    """
    Check if target_int can be formed by a sum of a non-empty subset of weights.
    weights are positive integers.
    Uses meet-in-the-middle for efficiency when len(weights) is moderate.
    """
    n = len(weights)
    if n == 0:
        return False
    
    # Quick checks
    total = sum(weights)
    if target_int > total:
        return False
    if target_int == 0:
        return False # Non-empty subset required, weights are positive, so sum > 0
        
    # If target is exactly one of the weights
    if target_int in weights:
        return True

    if n <= 20:
        # Meet in the middle
        half = n // 2
        w1 = weights[:half]
        w2 = weights[half:]
        
        # Generate all subset sums for w1
        sums1 = set([0])
        for w in w1:
            new_sums = set()
            for s in sums1:
                ns = s + w
                if ns <= target_int: # Pruning
                    new_sums.add(ns)
            sums1.update(new_sums)
            
        # Generate all subset sums for w2
        sums2 = set([0])
        for w in w2:
            new_sums = set()
            for s in sums2:
                ns = s + w
                if ns <= target_int: # Pruning
                    new_sums.add(ns)
            sums2.update(new_sums)
            
        # Check if any pair sums to target
        # target = s1 + s2 => s2 = target - s1
        for s1 in sums1:
            if (target_int - s1) in sums2:
                # Ensure non-empty subset
                # If s1==0 and s2==0, subset is empty. 
                # But weights are positive, so if target > 0, we can't have s1=0 and s2=0.
                # Wait, if target > 0, s1+s2=target implies at least one is non-zero.
                # So the subset is non-empty.
                return True
        return False
    else:
        # For larger n, use a simple greedy/backtracking or just return False if we can't prove?
        # Actually, for n > 20, MITM is 2^10 * 2^10 = 1M ops per check.
        # If n is up to 24, 2^12 * 2^12 = 16M ops. This might be slow if called many times.
        # But in our construction, n is the size of the current set, which grows.
        # Let's stick to MITM but optimize.
        # Actually, 2^12 = 4096. 4096*4096 = 16M. In Python, this is ~1-2 seconds.
        # If we call this 100 times, it's too slow.
        # We need to be careful.
        
        # Alternative: Since we are building the set, n is small initially.
        # As n grows, the checks become more expensive.
        # However, the number of candidates we test is also bounded.
        
        # Let's use a slightly more optimized MITM or just standard DFS with pruning for n > 20?
        # Or, better: since weights are derived from 1/x, they have specific structure.
        # But generic MITM is safe.
        
        # Let's cap the MITM size. If n > 24, it's too big anyway (max 24 elements).
        # So n <= 24.
        # Max MITM cost is 2^12 * 2^12 = 16M set lookups.
        # This is acceptable for a few calls, but not for thousands.
        # We will use this in _can_add.
        
        # Re-implement with better pruning
        sums1 = set([0])
        for w in w1:
            new_sums = set()
            for s in sums1:
                ns = s + w
                if ns <= target_int:
                    new_sums.add(ns)
            sums1.update(new_sums)
            
        sums2 = set([0])
        for w in w2:
            new_sums = set()
            for s in sums2:
                ns = s + w
                if ns <= target_int:
                    new_sums.add(ns)
            sums2.update(new_sums)
            
        for s1 in sums1:
            if (target_int - s1) in sums2:
                return True
        return False

def _can_add(k, selected):
    """
    Check if k can be added to selected set S.
    Condition:
    1. 1/k is not a subsum of S.
    2. For all x in S, 1/x is not a subsum of (S \ {x}) U {k}.
       This implies 1/x != sum(T) + 1/k for any T subset of S\{x}.
       => 1/x - 1/k != sum(T) for any T subset of S\{x}.
       => If 1/x > 1/k, check if (1/x - 1/k) is a subsum of S\{x}.
    """
    if not selected:
        return True
        
    # Scale factor for S
    L_S = _lcm_list(selected)
    
    # Check 1: Is 1/k a subsum of S?
    # 1/k = sum(1/y for y in T) => L_S/k = sum(L_S/y for y in T)
    target_1 = L_S // k
    weights_1 = [L_S // y for y in selected]
    if _is_subset_sum_integer(target_1, weights_1):
        return False
        
    # Check 2: For each x in S
    for x in selected:
        if x >= k:
            # 1/x <= 1/k, so 1/x - 1/k <= 0. Sum of positive reciprocals > 0.
            # So 1/x - 1/k cannot be a subsum.
            continue
            
        # 1/x > 1/k
        # We need to check if 1/x - 1/k is a subsum of S \ {x}.
        # Let S' = S \ {x}.
        # L_S' = LCM(S')
        # Note: L_S = LCM(S) = LCM(S' U {x}) = LCM(L_S', x).
        # So L_S is a multiple of L_S'.
        
        S_prime = [y for y in selected if y != x]
        if not S_prime:
            continue # 1/x - 1/k cannot be 0 (x!=k) and sum of empty set is 0.
            
        L_S_prime = _lcm_list(S_prime)
        
        # Target: 1/x - 1/k = (k - x) / (x * k)
        # We want to check if (k-x)/(x*k) is a subsum of {1/y for y in S_prime}.
        # Multiply by L_S_prime:
        # Target_int = L_S_prime * (k - x) / (x * k)
        # This must be an integer.
        # Since L_S_prime is a multiple of all y in S_prime, but not necessarily x or k.
        # Wait, the subsum equation is:
        # sum(1/y for y in T) = (k-x)/(x*k)
        # Multiply by L_S_prime:
        # sum(L_S_prime/y for y in T) = L_S_prime * (k-x) / (x*k)
        # The LHS is an integer. So the RHS must be an integer.
        # If RHS is not an integer, the equation has no solution in integers, so no subsum.
        
        numerator = L_S_prime * (k - x)
        denominator = x * k
        
        if numerator % denominator != 0:
            continue # Not an integer, so impossible
            
        target_2 = numerator // denominator
        weights_2 = [L_S_prime // y for y in S_prime]
        
        if _is_subset_sum_integer(target_2, weights_2):
            return False
            
    return True

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    """
    S = []
    # Try different orders?
    # Descending seems good for speed (large numbers are safe).
    # Let's try descending first.
    candidates = list(range(24, 0, -1))
    
    for c in candidates:
        if _time_remaining() < 5.0:
            break
        if _can_add(c, S):
            S.append(c)
            
    # Local search / Hill climbing?
    # Try to add any remaining number?
    # The greedy loop above already tried all.
    # But maybe a different order yields a larger set?
    # Let's try a few random permutations if time allows.
    
    best_set = S
    best_size = len(S)
    
    # Try a few random restarts with different orderings
    import random
    random.seed(42)
    
    attempts = 0
    while _time_remaining() > 10.0 and attempts < 50:
        attempts += 1
        perm = list(range(1, 25))
        random.shuffle(perm)
        S_curr = []
        for c in perm:
            if _time_remaining() < 5.0:
                break
            if _can_add(c, S_curr):
                S_curr.append(c)
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr
            # If we hit 24, we're done (unlikely but possible)
            if best_size == 24:
                break
                
    return best_set

# Quick self-test
if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}, Size: {len(res)}")
    
    # Verify
    def verify(S):
        S = set(S)
        for x in S:
            others = [y for y in S if y != x]
            # Check if 1/x is a subsum of others
            # Use integer check
            if not others:
                continue
            L = _lcm_list(others)
            target = L // x
            weights = [L // y for y in others]
            if _is_subset_sum_integer(target, weights):
                return False
        return True
        
    print(f"Valid: {verify(res)}")
