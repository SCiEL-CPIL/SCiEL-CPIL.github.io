import time
import random

def construct():
    deadline = time.time() + 85.0
    best_set = []
    
    def can_add(x, S, sums_set):
        n = len(S)
        if n == 0:
            return True, {3*x}
            
        new_sums = set()
        
        # 1. (x, x, x)
        s = 3 * x
        if s in sums_set or s in new_sums:
            return False, None
        new_sums.add(s)
        
        # 2. (x, x, y)
        for y in S:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # 3. (x, y, y)
        for y in S:
            s = x + 2 * y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # 4. (x, y, z)
        # Optimization: Precompute pair sums? No, S changes.
        # Just nested loops.
        for i in range(n):
            y = S[i]
            for j in range(i, n):
                z = S[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
                
        return True, new_sums

    # Randomized Greedy with Restarts
    while time.time() < deadline:
        current_set = []
        sums_set = set()
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            ok, new_sums = can_add(x, current_set, sums_set)
            if ok:
                current_set.append(x)
                sums_set.update(new_sums)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
