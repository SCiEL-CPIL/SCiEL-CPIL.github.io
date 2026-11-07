import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    A = list(range(17, 33))
    B = list(range(65, 129))
    seed_S = A + B
    best_S = list(seed_S)
    best_size = len(best_S)
    
    def can_add_fast(S, S_set, x):
        # S does not contain x
        # Find all v in S such that x+v in S
        # These are the only candidates for u and v in the violating triple
        candidates = set()
        for s in S:
            val = s - x
            if val > 0 and val in S_set:
                candidates.add(val)
        
        # Now check pairs u, v from candidates
        # u, v must be distinct
        # and u+v in S or u+v == x
        c_list = list(candidates)
        n = len(c_list)
        for i in range(n):
            u = c_list[i]
            for j in range(i+1, n):
                v = c_list[j]
                if u + v in S_set or u + v == x:
                    return False
        return True

    def greedy_add(initial_S):
        S = list(initial_S)
        S_set = set(S)
        candidates = [i for i in range(1, 129) if i not in S_set]
        random.shuffle(candidates)
        
        while time.time() < deadline:
            added = False
            for x in candidates:
                if x in S_set:
                    continue
                if can_add_fast(S, S_set, x):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    added = True
                    break
            if not added:
                break
        return S

    # Initial run
    best_S = greedy_add(seed_S)
    best_size = len(best_S)
    
    # Multiple restarts
    while time.time() < deadline:
        S = greedy_add(seed_S)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    return best_S
