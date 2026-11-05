import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0
    
    cap = 1000
    best_set = set()
    best_len = 0
    rng = random.Random(42)
    
    def greedy_build(order):
        chosen = []
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        chosen_set = set()
        
        for x in order:
            if x in chosen_set:
                continue
            # Check if x can be added
            # Sums: 2x, x+y for y in chosen
            valid = True
            if counts[2*x] >= 2:
                valid = False
            else:
                # Check x+y for all y in chosen
                # We can do this in a loop or vectorized
                # Loop is probably faster for small sets, but vectorized is better for large
                # Let's try a quick loop first, if set is small
                for y in chosen:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
                if valid:
                    # Also need to check if 2x is already 2? 
                    # The loop above doesn't check 2x if y=x is not in chosen.
                    # But we checked counts[2*x] >= 2 above.
                    # Wait, if x is added, counts[2*x] increases by 1.
                    # If counts[2*x] was 1, it becomes 2 (OK).
                    # If counts[2*x] was 2, it becomes 3 (Invalid).
                    # So checking counts[2*x] < 2 is correct.
                    pass
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                counts[2*x] += 1
                for y in chosen[:-1]:
                    counts[x + y] += 1
        
        return chosen, counts, chosen_set

    def try_add(set_obj, counts, x):
        if x in set_obj:
            return False
        if counts[2*x] >= 2:
            return False
        for y in set_obj:
            if y == x: continue
            if counts[x + y] >= 2:
                return False
        return True

    def do_add(set_obj, counts, x):
        set_obj.add(x)
        counts[2*x] += 1
        for y in set_obj:
            if y != x:
                counts[x + y] += 1

    def do_remove(set_obj, counts, x):
        set_obj.remove(x)
        counts[2*x] -= 1
        for y in set_obj:
            if y != x:
                counts[x + y] -= 1

    def is_sum_involving_u(s, u, set_obj):
        if s == 2 * u:
            return True
        if s > u:
            v = s - u
            if v in set_obj and v != u:
                return True
        return False

    while _time_left() > 5.0:
        # 1. Greedy Init
        order = list(range(1, cap + 1))
        rng.shuffle(order)
        current_set, counts, current_set_obj = greedy_build(order)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = set(current_set)
            
        # 2. Local Search
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            # Try to add
            for x in range(1, cap + 1):
                if x not in current_set_obj:
                    if try_add(current_set_obj, counts, x):
                        do_add(current_set_obj, counts, x)
                        improved = True
                        if len(current_set_obj) > best_len:
                            best_len = len(current_set_obj)
                            best_set = set(current_set_obj)
            
            # Try swaps
            # To avoid O(N^2 * |S|) being too slow, we can sample or limit
            # But N=1000, |S|~500. 1000*500*500 = 250M. Too slow in Python.
            # We need to optimize the swap check.
            
            # Optimization: 
            # For a fixed u, we want to find v such that swap is valid.
            # Condition: for all y in S\{u}, counts[v+y] - is_sum_involving_u(v+y, u, S) < 2
            # i.e., counts[v+y] <= 1 + is_sum_involving_u(v+y, u, S)
            
            # Let's try a limited number of random swaps per iteration to keep it fast
            # Or just iterate u, and pick random v?
            # Better: iterate all v, but break early if possible?
            
            # Let's try to be smart:
            # If we swap u for v, the "new" sums are v+y.
            # The "old" sums removed are u+y.
            
            # We'll do a pass over all u in S. For each u, we'll try a few random v not in S.
            # This reduces the cost significantly.
            
            list_S = list(current_set_obj)
            list_not_S = [x for x in range(1, cap+1) if x not in current_set_obj]
            
            if not list_not_S:
                break
                
            for u in list_S:
                if _time_left() < 5.0:
                    break
                # Try a few random v's
                for _ in range(5): # 5 random candidates per u
                    if not list_not_S:
                        break
                    v_idx = rng.randint(0, len(list_not_S)-1)
                    v = list_not_S[v_idx]
                    
                    # Check if swap u->v is valid
                    valid = True
                    # Check 2v
                    if counts[2*v] >= 2 and not is_sum_involving_u(2*v, u, current_set_obj):
                        valid = False
                    else:
                        # Check v+y for y in S\{u}
                        # We can skip if counts[v+y] is low
                        for y in current_set_obj:
                            if y == u:
                                continue
                            s = v + y
                            c = counts[s]
                            if c >= 2:
                                if not is_sum_involving_u(s, u, current_set_obj):
                                    valid = False
                                    break
                            # If c < 2, it's definitely valid for this sum
                            # because c - 0 <= 1 is always true if c<2.
                            # If c==2, we need is_sum_involving_u to be True.
                            # If c>2, it's invalid regardless? 
                            # Wait. Max count is 2. So c can be 0, 1, 2.
                            # If c==2, we need is_sum_involving_u(s,u,S) to be True.
                            # If c<2, it's fine.
            
                    if valid:
                        # Perform swap
                        do_remove(current_set_obj, counts, u)
                        do_add(current_set_obj, counts, v)
                        # Remove v from list_not_S
                        list_not_S.pop(v_idx)
                        improved = True
                        # Update best if size increased? No, size same.
                        # But structure changed.
                        # We don't update best_set here unless we add later.
                        
        # End local search for this run

    return list(best_set)

