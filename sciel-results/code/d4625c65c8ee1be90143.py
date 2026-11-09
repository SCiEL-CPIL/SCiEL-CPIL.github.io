import time
import random
import sys

def construct():
    """
    Returns a list of up to 48 distinct positive integers <= 512 such that no element 
    is the exact arithmetic mean of any subset of at least two other elements.
    
    Strategy:
    1. Generate 3-AP-free sets using optimized greedy.
    2. Validate full condition.
    3. Repair invalid sets by removing mean-elements.
    """
    deadline = time.time() + 85
    N = 512
    target_size = 48
    
    def is_valid_full(S):
        m = len(S)
        if m < 3:
            return True, []
        
        S_sorted = sorted(S)
        violations = []
        
        for idx, a in enumerate(S_sorted):
            max_k = m - 1
            dp = [0] * (max_k + 1)
            dp[0] = 1
            
            count = 0
            for v in S_sorted:
                if v == a and count == 0:
                    count = 1
                    continue
                
                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << v
                count += 1
            
            for k in range(2, max_k + 1):
                target_sum = a * k
                if target_sum > 512 * k:
                    continue
                if target_sum > 24576:
                    break
                if (dp[k] >> target_sum) & 1:
                    violations.append((a, k))
                    break
            
        return len(violations) == 0, violations

    def greedy_3ap_free(order):
        S = []
        S_set = set()
        
        for c in order:
            if c in S_set:
                continue
            
            conflict = False
            for a in S:
                # c is mean of a and b => b = 2c - a
                b1 = 2 * c - a
                if b1 in S_set and b1 != a:
                    conflict = True
                    break
                
                # a is mean of c and b => b = 2a - c
                b2 = 2 * a - c
                if b2 in S_set and b2 != c:
                    conflict = True
                    break
                    
            if not conflict:
                S.append(c)
                S_set.add(c)
                
        return S

    def repair_set(S):
        S = S[:]
        S_set = set(S)
        # Limit repair iterations to avoid infinite loops or excessive time
        for _ in range(10):
            if not S:
                break
            valid, violations = is_valid_full(S)
            if valid:
                break
            a, k = violations[0]
            S.remove(a)
            S_set.discard(a)
        return S

    best_set = []
    best_len = 0
    rng = random.Random(42)
    
    # Phase 1: Deterministic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)),
        list(range(2, N + 1, 2)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_3ap_free(order)
        valid, _ = is_valid_full(s)
        if valid:
            if len(s) > best_len:
                best_len = len(s)
                best_set = s[:]
        else:
            repaired = repair_set(s)
            if len(repaired) > best_len:
                best_len = len(repaired)
                best_set = repaired[:]
        
        if best_len == target_size:
            return best_set

    # Phase 2: Randomized
    for i in range(200):
        if time.time() > deadline:
            break
        p = list(range(1, N + 1))
        rng.shuffle(p)
        s = greedy_3ap_free(p)
        valid, _ = is_valid_full(s)
        if valid:
            if len(s) > best_len:
                best_len = len(s)
                best_set = s[:]
        else:
            repaired = repair_set(s)
            if len(repaired) > best_len:
                best_len = len(repaired)
                best_set = repaired[:]
        
        if best_len == target_size:
            return best_set

    return best_set

# EVOLVE-BLOCK-START
# The code above is the full implementation.
# EVOLVE-BLOCK-END
