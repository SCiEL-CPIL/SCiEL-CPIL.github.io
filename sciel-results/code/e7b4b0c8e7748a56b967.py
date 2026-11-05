import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 192
    
    def can_add(x, S):
        """S is a list of selected elements. Check if x can be added."""
        m = len(S)
        if m < 3:
            if m >= 2:
                for a in S:
                    others = [v for v in S if v != a]
                    for i in range(len(others)):
                        for j in range(i+1, len(others)):
                            if (x + others[i] + others[j]) % a == 0:
                                return False
            return True
        
        S_arr = np.array(S)
        
        # Check 1: x does not divide any triple sum in S
        S_mod_x = S_arr % x
        for i in range(m):
            for j in range(i+1, m):
                target = (-(S[i] + S[j])) % x
                mask = S_mod_x == target
                mask[i] = False
                mask[j] = False
                if mask.any():
                    return False
        
        # Check 2: for each a in S, a does not divide x + b + c for any pair (b,c) in S\{a}
        for idx_a, a in enumerate(S):
            others_idx = [i for i in range(m) if i != idx_a]
            if len(others_idx) < 2:
                continue
            others_arr = S_arr[others_idx]
            target = (-x) % a
            others_mod_a = others_arr % a
            for i in range(len(others_idx)):
                for j in range(i+1, len(others_idx)):
                    if (others_mod_a[i] + others_mod_a[j]) % a == target:
                        return False
        
        return True
    
    best_size = 0
    best_set = []
    
    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
        return S
    
    orders = []
    orders.append(list(range(n, 0, -1)))
    for _ in range(15):
        o = list(range(1, n+1))
        random.shuffle(o)
        orders.append(o)
    for _ in range(15):
        large = list(range(n, n//2, -1))
        small = list(range(n//2, 0, -1))
        random.shuffle(large)
        random.shuffle(small)
        orders.append(large + small)
    
    for order in orders:
        if time.time() > deadline - 3:
            break
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            if best_size >= 48:
                break
    
    if best_set and time.time() < deadline - 3:
        S = list(best_set)
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for x in range(1, n+1):
                if x in S_set:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    return best_set
