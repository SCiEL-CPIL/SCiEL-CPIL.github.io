import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0  # Safety margin from 90s limit
    
    def is_valid_addition(S, x):
        """
        Fast incremental check: can x be added to valid set S?
        1. x not mean of any subset of S (size>=2)
        2. No a in S becomes mean of a new subset containing x
        """
        n = len(S)
        if n < 2:
            return True
        
        # Check 1: x not mean of subset of S
        max_sum_S = sum(S)
        dp_S = [0] * (n + 1)
        dp_S[0] = 1
        for a in S:
            for k in range(n - 1, -1, -1):
                if dp_S[k]:
                    dp_S[k+1] |= (dp_S[k] << a)
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum_S:
                break
            if (dp_S[k] >> target) & 1:
                return False
        
        # Check 2: For each a, no new subset with x makes a a mean
        for i, a in enumerate(S):
            others_no_a = S[:i] + S[i+1:]
            m = len(others_no_a)
            if m < 1:
                continue
            max_sum = sum(others_no_a)
            dp_o = [0] * (m + 1)
            dp_o[0] = 1
            for b in others_no_a:
                for k in range(m - 1, -1, -1):
                    if dp_o[k]:
                        dp_o[k+1] |= (dp_o[k] << b)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if target < 0 or target > max_sum:
                    continue
                if (dp_o[j] >> target) & 1:
                    return False
        return True

    best_set = []
    best_size = 0
    
    # Phase 1: Greedy with various orders
    base = list(range(1, 513))
    orders = [base, base[::-1]]
    
    random.seed(42)
    for _ in range(100):
        shuffled = base[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        current_set = []
        for x in order:
            if len(current_set) >= 48:
                break
            if is_valid_addition(current_set, x):
                current_set.append(x)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        if best_size == 48:
            return best_set

    # Phase 2: Local Search
    if best_size < 48 and time.time() < deadline - 5:
        S = list(best_set)
        S_set = set(S)
        unused = [x for x in range(1, 513) if x not in S_set]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u in unused:
                if len(S) >= 48:
                    break
                if is_valid_addition(S, u):
                    S.append(u)
                    S_set.add(u)
                    unused.remove(u)
                    improved = True
                    break
            if len(S) >= 48:
                break
            random.shuffle(unused)
            for u in unused[:100]:
                if time.time() > deadline - 2:
                    break
                for i, s in enumerate(S):
                    if is_valid_addition(S[:i] + S[i+1:], u):
                        S[i] = u
                        S_set.remove(s)
                        S_set.add(u)
                        unused.remove(u)
                        unused.append(s)
                        improved = True
                        break
                if improved:
                    break
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    return best_set
