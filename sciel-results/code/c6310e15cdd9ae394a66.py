import time
import random
import numpy as np

def is_valid(S):
    if len(S) < 3:
        return True
    lst = np.array(sorted(S), dtype=np.int64)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        rest = lst[i+1:]
        if len(rest) == 0:
            continue
        # Check if any pair sum from rest is divisible by a
        # We can optimize by checking only upper triangle if needed, but full is fine
        sums = rest[:, None] + rest[None, :]
        if np.any(sums % a == 0):
            return False
    return True

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Initial Greedy Solution (Descending)
    S = set()
    for a in range(n, 0, -1):
        valid = True
        lst = list(S)
        m = len(lst)
        for i in range(m):
            b = lst[i]
            for j in range(i, m):
                c = lst[j]
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    # 2. Local Search
    available = [x for x in range(1, n + 1) if x not in S]
    
    while time.time() < deadline:
        if not available:
            break
            
        random.shuffle(available)
        
        added = False
        for y in available:
            new_S = S.union({y})
            if is_valid(new_S):
                S = new_S
                available.remove(y)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                added = True
                break
        
        if added:
            continue
            
        # Try swap
        if not S:
            break
            
        x_to_remove = random.choice(list(S))
        y_to_add = random.choice(available)
        
        new_S = S.copy()
        new_S.remove(x_to_remove)
        new_S.add(y_to_add)
        
        if is_valid(new_S):
            S = new_S
            available.remove(y_to_add)
            available.append(x_to_remove)
            
    return list(best_S)
