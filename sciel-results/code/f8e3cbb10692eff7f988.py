import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0
    n_max = 512
    target_size = 48

    def is_valid_set(S):
        """Full verification: no element is mean of any subset of size >= 2 of others."""
        S_set = set(S)
        for x in S:
            others = [s for s in S if s != x]
            n = len(others)
            if n < 2:
                continue
            max_sum = sum(others)
            max_k = min(n, max_sum // x)
            if max_k < 2:
                continue
            dp = [0] * (n + 1)
            dp[0] = 1
            for val in others:
                for k in range(n - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << val)
            for k in range(2, max_k + 1):
                target = k * x
                if (dp[k] >> target) & 1:
                    return False
        return True

    def fast_conflict_check(x, S):
        """Check if x conflicts with S."""
        S_set = set(S)
        n = len(S)
        if n < 2:
            for a in S:
                others_no_a = [s for s in S if s != a]
                m = len(others_no_a)
                if m >= 1:
                    max_sum_m = sum(others_no_a)
                    max_j = min(m, max_sum_m // max(1, a))
                    dp_m = [0] * (m + 1)
                    dp_m[0] = 1
                    for val in others_no_a:
                        for k in range(m - 1, -1, -1):
                            if dp_m[k]:
                                dp_m[k+1] |= (dp_m[k] << val)
                    for j in range(1, m + 1):
                        target_sum = (j + 1) * a - x
                        if 0 <= target_sum <= max_sum_m:
                            if (dp_m[j] >> target_sum) & 1:
                                return False
            return True
        
        max_sum = sum(S)
        max_k = min(n, max_sum // x)
        if max_k >= 2:
            dp = [0] * (n + 1)
            dp[0] = 1
            for val in S:
                for k in range(n - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << val)
            for k in range(2, max_k + 1):
                target = k * x
                if (dp[k] >> target) & 1:
                    return False
        
        for a in S:
            others_no_a = [s for s in S if s != a]
            m = len(others_no_a)
            if m >= 1:
                max_sum_m = sum(others_no_a)
                max_j = min(m, max_sum_m // max(1, a))
                dp_m = [0] * (m + 1)
                dp_m[0] = 1
                for val in others_no_a:
                    for k in range(m - 1, -1, -1):
                        if dp_m[k]:
                            dp_m[k+1] |= (dp_m[k] << val)
                for j in range(1, m + 1):
                    target_sum = (j + 1) * a - x
                    if 0 <= target_sum <= max_sum_m:
                        if (dp_m[j] >> target_sum) & 1:
                            return False
        return True

    rng = random.Random(12345)
    best_set = []
    best_size = 0

    # Modular class orderings + random
    base_orders = [
        list(range(n_max, 0, -1)),
        list(range(1, n_max + 1)),
    ]
    
    for mod in [5, 7, 11]:
        for r in range(mod):
            base_orders.append([x for x in range(1, n_max + 1) if x % mod == r])
    
    for i in range(40):
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        base_orders.append(order)

    for order in base_orders:
        if time.time() > deadline:
            break
        S = []
        S_set = set()
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if fast_conflict_check(x, S):
                S.append(x)
                S_set.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Local search
    if best_size < target_size and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            for x in available:
                if len(S) >= target_size:
                    break
                if fast_conflict_check(x, S):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= target_size:
                break
            
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    old = S[i]
                    S_temp = S[:i] + S[i+1:]
                    
                    for x in available:
                        if x in S_set:
                            continue
                        if fast_conflict_check(x, S_temp):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(old)
                            improved = True
                            break
                    if improved:
                        break

    # Final verification
    if best_set and is_valid_set(best_set):
        return best_set
    elif S and is_valid_set(S):
        return S
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
