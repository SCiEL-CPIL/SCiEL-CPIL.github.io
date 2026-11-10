import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_set = set()
    best_size = 0
    
    def is_valid_insertion(S, counts, x):
        if counts.get(2*x, 0) >= 2:
            return False
        for y in S:
            if y == x: continue
            if counts.get(x + y, 0) >= 2:
                return False
        return True

    def add_to_state(S, counts, x):
        S.add(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in S:
            if y != x:
                s = x + y
                counts[s] = counts.get(s, 0) + 1

    def remove_from_state(S, counts, x):
        S.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in S:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    def is_valid_swap(S, counts, x_remove, x_add):
        if x_remove == x_add:
            return False
        
        # Check sum 2 * x_add
        s = 2 * x_add
        removed = 0
        if s == 2 * x_remove:
            removed += 1
        y_cand = s - x_remove
        if y_cand in S and y_cand != x_remove:
            removed += 1
        if counts.get(s, 0) - removed > 1:
            return False
            
        # Check sums x_add + y
        for y in S:
            if y == x_remove:
                continue
            s = x_add + y
            removed = 0
            if s == 2 * x_remove:
                removed += 1
            y_cand = s - x_remove
            if y_cand in S and y_cand != x_remove:
                removed += 1
            if counts.get(s, 0) - removed > 1:
                return False
        return True

    def local_search(S, counts):
        nonlocal best_set, best_size
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try Insertions
            candidates = [x for x in range(1, 1001) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_insertion(S, counts, x):
                    add_to_state(S, counts, x)
                    improved = True
                    break
            
            if not improved:
                # Try Swaps
                swaps_tried = 0
                max_swaps = 50
                while swaps_tried < max_swaps and time.time() < deadline:
                    if not S: break
                    x_remove = random.choice(list(S))
                    x_add = random.randint(1, 1000)
                    while x_add in S:
                        x_add = random.randint(1, 1000)
                    
                    if is_valid_swap(S, counts, x_remove, x_add):
                        remove_from_state(S, counts, x_remove)
                        add_to_state(S, counts, x_add)
                        improved = True
                        break
                    swaps_tried += 1
                    
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # 1. Ascending Greedy
    S1 = set(); C1 = {}
    for x in range(1, 1001):
        if is_valid_insertion(S1, C1, x):
            add_to_state(S1, C1, x)
    if len(S1) > best_size:
        best_size = len(S1); best_set = set(S1)
    local_search(S1, C1)

    # 2. Descending Greedy
    S2 = set(); C2 = {}
    for x in range(1000, 0, -1):
        if is_valid_insertion(S2, C2, x):
            add_to_state(S2, C2, x)
    if len(S2) > best_size:
        best_size = len(S2); best_set = set(S2)
    local_search(S2, C2)

    # 3. Random Restart Local Search
    while time.time() < deadline:
        S3 = set(); C3 = {}
        nums = list(range(1, 1001))
        random.shuffle(nums)
        for x in nums:
            if is_valid_insertion(S3, C3, x):
                add_to_state(S3, C3, x)
        
        if len(S3) > best_size:
            best_size = len(S3); best_set = set(S3)
        
        local_search(S3, C3)

    return list(best_set)
