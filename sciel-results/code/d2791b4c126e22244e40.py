import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute conflict_map
    conflict_map = {x: [] for x in nums}
    
    for a in nums:
        for b in nums:
            if a >= b:
                continue
            for c in nums:
                if b >= c:
                    continue
                if gcd(a, b) == gcd(b, c) == gcd(c, a):
                    conflict_map[a].append((b, c))
                    conflict_map[b].append((a, c))
                    conflict_map[c].append((a, b))
    
    def can_add(x, S):
        for a, b in conflict_map[x]:
            if a in S and b in S:
                return False
        return True

    def greedy_init():
        S = set()
        order = nums.copy()
        random.shuffle(order)
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    best_set = greedy_init()
    best_size = len(best_set)
    
    swap_count = 0
    max_swaps_per_restart = 1000
    
    while time.time() < deadline:
        current_set = set(best_set)
        
        # Try to add
        candidates = [x for x in nums if x not in current_set]
        random.shuffle(candidates)
        
        added = False
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
                added = True
                break
        
        if added:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
            swap_count = 0
        else:
            # Try to swap
            if candidates and current_set:
                y = random.choice(list(current_set))
                x = random.choice(candidates)
                
                current_set.remove(y)
                if can_add(x, current_set):
                    current_set.add(x)
                    swap_count += 1
                else:
                    current_set.add(y)
                    swap_count += 1
                    
            if swap_count > max_swaps_per_restart:
                # Restart with new greedy init
                new_set = greedy_init()
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
                swap_count = 0
                # Continue with the new best set in next iteration
                
    return best_set
