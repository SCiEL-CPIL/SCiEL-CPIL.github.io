import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def can_add(S, x):
        if x in S:
            return True
        S_list = sorted(S)
        m = len(S_list)
        
        # Case 1: a=x, b,c in S, x < b <= c, check x ∤ (b+c)
        for i in range(m):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, m):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: b=x, a,c in S, a < x <= c, check a ∤ (x+c)
        for i in range(m):
            a = S_list[i]
            if a >= x:
                break
            for j in range(m):
                c = S_list[j]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: c=x, a,b in S, a < b <= x, check a ∤ (b+x)
        for i in range(m):
            a = S_list[i]
            if a >= x:
                break
            for j in range(i + 1, m):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    k = 171
    S = set(range(k, n + 1))
    
    # Greedy add from below
    for x in range(k - 1, 0, -1):
        if time.time() > deadline - 25:
            break
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    random.seed(42)
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            S = set(best_S)
        else:
            S = set(best_S)
            S_list = list(S)
            num_remove = min(5, max(1, len(S_list) // 10))
            for _ in range(num_remove):
                if S_list:
                    S.remove(random.choice(S_list))
                    S_list = list(S)
        
        # Greedy add
        candidates = [x for x in range(1, n + 1) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 15:
                break
            if can_add(S, x):
                S.add(x)
        
        # Local improvement
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for remove_x in S_list[:25]:
                if time.time() > deadline - 15:
                    break
                remaining = set(S)
                remaining.discard(remove_x)
                
                candidates = [x for x in range(1, n + 1) if x not in remaining]
                random.shuffle(candidates)
                
                for add_y in candidates[:50]:
                    if can_add(remaining, add_y):
                        S = remaining
                        S.add(add_y)
                        improved = True
                        break
                if improved:
                    break
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
    
    return list(best_S)
