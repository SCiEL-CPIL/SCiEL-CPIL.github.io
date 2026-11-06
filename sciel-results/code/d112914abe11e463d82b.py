import random
import time

def construct():
    n = 4096
    best_set = []
    
    # Helper to check if x can be added to current_set
    def can_add(x, current_set, existing_sums):
        # Generate new sums involving x
        # (x, x, x)
        s = 3 * x
        if s in existing_sums:
            return False, []
        new_sums = [s]
        
        # (x, x, y) for y in current_set
        for y in current_set:
            s = 2 * x + y
            if s in existing_sums:
                return False, []
            new_sums.append(s)
            
        # (x, y, z) for y, z in current_set
        # y, z can be same. y <= z to avoid duplicate generation? 
        # Actually, sum is commutative. We just need to check if sum exists.
        # We don't need to avoid generating duplicate sums in new_sums if we just check against existing_sums.
        # But we must ensure we don't miss any.
        # Iterating all pairs (y, z) is O(k^2).
        for i in range(len(current_set)):
            for j in range(i, len(current_set)): # y <= z
                y = current_set[i]
                z = current_set[j]
                s = x + y + z
                if s in existing_sums:
                    return False, []
                new_sums.append(s)
                
        return True, new_sums

    def run_greedy(order):
        current_set = []
        existing_sums = set()
        for x in order:
            ok, new_sums = can_add(x, current_set, existing_sums)
            if ok:
                current_set.append(x)
                existing_sums.update(new_sums)
        return current_set

    # Strategy 1: Random Shuffles
    best_size = 0
    for _ in range(50): # 50 attempts
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        res = run_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    # Strategy 2: Largest First
    candidates_desc = list(range(n, 0, -1))
    res = run_greedy(candidates_desc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res

    # Strategy 3: Smallest First (Original, but optimized)
    candidates_asc = list(range(1, n + 1))
    res = run_greedy(candidates_asc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res

    # Strategy 4: Random with bias towards larger numbers?
    # Or just more random shuffles.
    for _ in range(50):
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        res = run_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
    return best_set
