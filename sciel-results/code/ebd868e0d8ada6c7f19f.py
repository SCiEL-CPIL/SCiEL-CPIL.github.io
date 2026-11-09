import time
import random

def construct():
    deadline = time.time() + 55.0
    
    def subset_sum_dp(S):
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for val in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
        return dp

    def is_valid_add(S, x):
        # S is a list, x is int
        # Check 1: x is not mean of subset of S
        if S:
            dp_S = subset_sum_dp(S)
            n = len(S)
            for k in range(2, n + 1):
                target = k * x
                if (dp_S[k] >> target) & 1:
                    return False
        else:
            # S is empty, x is always valid
            pass

        # Check 2: For each a in S, a is not mean of subset of (S \ {a}) U {x} containing x
        for i, a in enumerate(S):
            R = S[:i] + S[i+1:]
            if not R:
                continue
            dp_R = subset_sum_dp(R)
            m = len(R)
            # k is size of subset in (S \ {a}) U {x}
            # Subset is {x} U T, T subset of R
            # Size k = 1 + |T|
            # Sum = x + sum(T)
            # Mean = a => x + sum(T) = k * a
            # sum(T) = k * a - x
            # |T| = k - 1
            # k >= 2 => |T| >= 1
            for k in range(2, m + 2):
                target = k * a - x
                if target < 0:
                    continue
                j = k - 1
                if j <= m:
                    if (dp_R[j] >> target) & 1:
                        return False
        return True

    best_set = []
    best_size = 0
    
    # Try a few strategies
    strategies = [
        list(range(512, 0, -1)),
        list(range(1, 513)),
        [x for x in range(1, 513) if x % 2 == 1],
        [x for x in range(1, 513) if x % 2 == 0]
    ]
    
    for seed in range(20):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        strategies.append(candidates)
        
    for candidates in strategies:
        if time.time() > deadline:
            break
        current = []
        S_set = set()
        
        for x in candidates:
            if len(current) >= 48:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if is_valid_add(current, x):
                current.append(x)
                S_set.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
            
        if best_size == 48:
            return best_set
            
    # If we didn't reach 48, try local search on the best set
    if best_set and best_size < 48 and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng = random.Random(time.time_ns())
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= 48:
                    break
                if is_valid_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
                    
            if len(S) >= 48:
                break
                
            # Try swapping
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    a = S[i]
                    S_temp = S[:i] + S[i+1:]
                    for x in available:
                        if is_valid_add(S_temp, x):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(a)
                            improved = True
                            break
                    if improved:
                        break
                        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set
