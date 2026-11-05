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
    
    Let U = S \ {a}. We need subset T ⊆ U with |T| >= 1 such that:
    x + sum(T) = (|T|+1) * a
    sum(T) = (|T|+1)*a - x
    
    Let w_t = t - a for t in U. Then sum(T) = sum_{t in T} w_t + |T|*a.
    So: sum_{t in T} w_t + |T|*a = (|T|+1)*a - x
    sum_{t in T} w_t = a - x
    
    So we need: does there exist a non-empty subset T ⊆ U such that sum_{t in T} (t-a) = a - x?
    """
    n = len(S)
    if n < 2:
        return False
    
    for a in S:
        U = [t for t in S if t != a]
        if len(U) < 1:
            continue
        
        target_val = a - x
        
        ws = [t - a for t in U]
        max_abs = sum(abs(w) for w in ws)
        if max_abs == 0:
            continue
        
        offset = max_abs
        mask = (1 << (2 * max_abs + 1)) - 1
        
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

def _sidon_seed(rng):
    """
    Generate a Sidon-like seed set: all pairwise sums distinct.
    This guarantees no element is the mean of any pair of others.
    """
    # Use a known construction: powers of 2 scaled, or greedy with pair-sum check
    # Greedy: pick elements ensuring all pairwise sums are unique
    S = []
    pair_sums = set()
    candidates = list(range(1, 513))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(S) >= 20:  # Start with ~20 seed elements
            break
        ok = True
        for y in S:
            s = x + y
            if s in pair_sums:
                ok = False
                break
        if ok:
            S.append(x)
            for y in S[:-1]:
                pair_sums.add(x + y)
            pair_sums.add(2 * x)  # x+x not needed for distinct elements, but safe
    
    return S

def construct():
    start_time = time.time()
    best_set = []
    best_len = 0
    
    rng = random.Random(42)
    
    # Multiple attempts with different strategies
    num_attempts = 50
    
    for attempt in range(num_attempts):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # Strategy 1: Sidon seed + greedy extension
        if attempt % 2 == 0:
            seed = _sidon_seed(rng)
            selected = seed[:]
        else:
            # Strategy 2: Pure greedy with random order
            selected = []
        
        # Build the order for extension
        remaining = [x for x in range(1, 513) if x not in selected]
        rng.shuffle(remaining)
        
        for x in remaining:
            if time.time() - start_time > _DEADLINE - 3:
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
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_set = selected[:]
            if best_len >= 48:
                break
    
    # Final verification
    if best_set:
        if not _full_check(best_set):
            return best_set
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
