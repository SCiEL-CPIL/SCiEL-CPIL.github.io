import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    n = 512
    
    # Precompute conflicts
    # conflict_sets[x] = list of tuples (y1, y2, y3) such that {x, y1, y2, y3} is a 4-AP
    conflict_sets = [[] for _ in range(n+1)]
    
    for a in range(1, n+1):
        for d in range(1, (n - a)//3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e > n: break
            conflict_sets[a].append((b, c, e))
            conflict_sets[b].append((a, c, e))
            conflict_sets[c].append((a, b, e))
            conflict_sets[e].append((a, b, c))

    def can_add(x, S):
        for others in conflict_sets[x]:
            if others[0] in S and others[1] in S and others[2] in S:
                return False
        return True

    # Initial set: base-4 digits 0,1
    def get_base4_set():
        S = set()
        for x in range(1, n+1):
            temp = x
            ok = True
            while temp > 0:
                if temp % 4 > 1:
                    ok = False
                    break
                temp //= 4
            if ok:
                S.add(x)
        return S

    best_set = get_base4_set()
    best_size = len(best_set)
    
    # Randomized Greedy
    while time.time() - start_time < time_limit:
        S = set(best_set) # Start from best found? Or from base4?
        # Starting from best found might get stuck.
        # Let's start from base4 each time, or mix.
        
        # Let's try starting from base4.
        S = get_base4_set()
        
        # Get candidates
        candidates = [x for x in range(1, n+1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return sorted(best_set)
