import numpy as np
import time
import random
from itertools import combinations

def _can_add(S, x):
    """
    Check if x can be added to set S without violating the mean condition.
    S is a list of already-accepted elements (not containing x).
    We need to ensure:
    1. x is not the mean of any subset of S of size >= 2.
    2. No element a in S is the mean of any subset of (S \ {a}) U {x} of size >= 2.
    
    For condition 1:
    x is mean of T (T subset of S, |T|>=2) iff sum(T) = |T| * x.
    Let d_t = t - x. Then sum(d_t) = 0 for some subset of size >= 2.
    
    For condition 2:
    a is mean of T' (T' subset of (S\{a}) U {x}, |T'|>=2) iff sum(T') = |T'| * a.
    Let d_t = t - a.
    If x not in T', then sum(T') = 0 for some subset of S\{a} of size >= 2.
    But we assume S is valid, so this is already false.
    If x in T', let T'' = T' \ {x}. Then T'' is a subset of S\{a}.
    sum(T'') + x = (|T''| + 1) * a
    sum(T'') = (|T''| + 1) * a - x
    sum(t - a for t in T'') = |T''| * a - x
    Let target = a - x.
    We need sum(t - a for t in T'') = |T''| * a - x? No.
    sum(T'') = sum(t for t in T'') = (|T''| + 1) * a - x.
    sum(t - a for t in T'') = sum(T'') - |T''| * a = (|T''| + 1) * a - x - |T''| * a = a - x.
    So we need a subset T'' of S\{a} such that sum(t - a for t in T'') = a - x.
    Also, the total size of the subset is |T''| + 1 (since x is included).
    We need |T''| + 1 >= 2, so |T''| >= 1.
    So we need a NON-EMPTY subset of S\{a} such that sum(t - a) = a - x.
    """
    # Condition 1: x is not mean of any subset of S of size >= 2
    if len(S) >= 2:
        diffs = [t - x for t in S]
        max_sum = sum(abs(d) for d in diffs)
        if max_sum == 0:
            return False
        offset = max_sum
        size = 2 * max_sum + 1
        INF = 1000
        dp = np.full(size, INF, dtype=np.int32)
        dp[offset] = 0
        
        for d in diffs:
            new_dp = dp.copy()
            if d > 0:
                new_dp[d:] = np.minimum(new_dp[d:], dp[:-d] + 1)
            elif d < 0:
                k = -d
                new_dp[:-k] = np.minimum(new_dp[:-k], dp[k:] + 1)
            dp = new_dp
            
            if dp[offset] > 0:
                return False

    # Condition 2: No a in S is mean of any subset involving x
    for a in S:
        others = [t for t in S if t != a]
        if not others:
            continue
            
        target = a - x
        diffs = [t - a for t in others]
        max_sum = sum(abs(d) for d in diffs)
        if max_sum == 0:
            continue
            
        offset = max_sum
        size = 2 * max_sum + 1
        INF = 1000
        dp = np.full(size, INF, dtype=np.int32)
        dp[offset] = 0
        
        target_idx = offset + target
        found = False
        
        for d in diffs:
            new_dp = dp.copy()
            if d > 0:
                new_dp[d:] = np.minimum(new_dp[d:], dp[:-d] + 1)
            elif d < 0:
                k = -d
                new_dp[:-k] = np.minimum(new_dp[:-k], dp[k:] + 1)
            dp = new_dp
            
            if 0 <= target_idx < size:
                if dp[target_idx] > 0:
                    found = True
                    break
        
        if found:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_set = []
    
    # Strategy 1: Greedy with different orders
    orders = []
    # Descending
    orders.append(list(range(512, 0, -1)))
    # Ascending
    orders.append(list(range(1, 513)))
    # Random shuffles
    random.seed(42)
    for _ in range(20):
        cands = list(range(1, 513))
        random.shuffle(cands)
        orders.append(cands)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S = []
        for x in order:
            if time.time() > deadline:
                break
                
            if _can_add(S, x):
                S.append(x)
                
        if len(S) > len(best_set):
            best_set = S
            
    # Strategy 2: Local search (swap)
    # Try to improve best_set by swapping elements
    # This is complex to implement efficiently with the current check.
    # Given time constraints, we might skip complex local search.
    
    return best_set

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
