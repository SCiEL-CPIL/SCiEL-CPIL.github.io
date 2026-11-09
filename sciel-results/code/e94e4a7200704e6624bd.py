import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    def run_one_trial():
        S = []
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)
        
        def try_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    return False
            S.append(x)
            in_S[x] = True
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1
            return True

        def try_remove(x):
            S.remove(x)
            in_S[x] = False
            counts[2 * x] -= 1
            for y in S:
                if y != x:
                    counts[x + y] -= 1

        # 1. Randomized Greedy Initialization
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            if try_add(c):
                pass
        
        current_size = len(S)
        
        # 2. Local Search with systematic blocker finding and random swaps
        # We try to add random elements not in S.
        # If blocked, find blockers and try swapping.
        
        while time.time() < deadline - 5:
            c = random.randint(1, max_val)
            if in_S[c]:
                continue
            
            # Try direct add
            if try_add(c):
                current_size += 1
                continue
            
            # If direct add failed, find blockers
            # Blockers are elements y in S such that removing y allows adding c.
            # Specifically, if counts[2*c] >= 2, blockers are pairs summing to 2*c.
            # If counts[c+y] >= 2, blockers are pairs summing to c+y.
            
            blockers_to_try = []
            
            # Check 2*c
            if counts[2 * c] >= 2:
                for a in S:
                    b = 2 * c - a
                    if b < a:
                        continue
                    if b > max_val:
                        continue
                    if in_S[b]:
                        blockers_to_try.append(a)
                        if b != a:
                            blockers_to_try.append(b)
            
            # Check c+y for y in S
            for y in S:
                s = c + y
                if counts[s] >= 2:
                    # Find the other pair u,v with u+v=s, u,v in S, u<=v
                    for u in S:
                        v = s - u
                        if v >= u and v <= max_val and in_S[v] and u != c and v != c:
                            blockers_to_try.append(u)
                            if v != u:
                                blockers_to_try.append(v)
                            break
            
            # Try swapping each blocker with c
            random.shuffle(blockers_to_try)
            swapped = False
            for blocker in blockers_to_try:
                if not in_S[blocker]:
                    continue
                try_remove(blocker)
                if try_add(c):
                    swapped = True
                    break
                else:
                    try_add(blocker)
            
            # If no swap worked, try random swaps as fallback
            if not swapped and len(S) > 0:
                for _ in range(3):
                    y = random.choice(S)
                    try_remove(y)
                    if try_add(c):
                        break
                    else:
                        try_add(y)

        return S

    # Run multiple trials if time permits
    best_S = run_one_trial()
    best_size = len(best_S)
    
    # If we have time, try another trial with different random seed
    if time.time() < deadline - 30:
        S2 = run_one_trial()
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
    
    return best_S
