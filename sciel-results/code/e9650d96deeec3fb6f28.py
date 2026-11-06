import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Greedy Initialization
    S = []
    counts = [0] * 2001  # Sums can be up to 2000
    
    for x in range(1, 1001):
        # Check if adding x is valid
        valid = True
        if counts[2*x] >= 2:
            valid = False
        else:
            for y in S:
                if counts[x+y] >= 2:
                    valid = False
                    break
        if valid:
            S.append(x)
            counts[2*x] += 1
            for y in S:
                if y != x: # y < x since S is sorted
                    counts[x+y] += 1
                else:
                    pass # 2x already handled
    
    # Local Search
    # We will try to add elements or swap elements
    # S is a list, but we need fast membership check? 
    # For "x not in S", a set is better.
    S_set = set(S)
    
    # Helper to check if x can be added to current S_set and counts
    def can_add(x):
        if x in S_set:
            return False
        if counts[2*x] >= 2:
            return False
        for y in S: # Iterating list is faster than set for small N?
            if counts[x+y] >= 2:
                return False
        return True

    # Helper to add x
    def add_x(x):
        S_set.add(x)
        S.append(x)
        counts[2*x] += 1
        for y in S:
            if y != x:
                counts[x+y] += 1
            else:
                pass

    # Helper to remove x
    def remove_x(x):
        S_set.discard(x)
        S.remove(x)
        counts[2*x] -= 1
        for y in S:
            if y != x:
                counts[x+y] -= 1
            else:
                pass

    best_size = len(S)
    
    while time.time() < deadline:
        # Phase 1: Try to add any element
        added = False
        # Random order for exploration
        candidates = [x for x in range(1, 1001) if x not in S_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x):
                add_x(x)
                added = True
                break
        
        if added:
            best_size = len(S)
            continue
        
        # Phase 2: If no addition, try a swap
        # Remove a random y, try to add a random x
        if len(S) == 0:
            break
            
        y = random.choice(S)
        remove_x(y)
        
        # Try to add any x not in S (which now includes y)
        # Note: y is not in S_set anymore
        candidates = [x for x in range(1, 1001) if x not in S_set]
        random.shuffle(candidates)
        
        swapped = False
        for x in candidates:
            if can_add(x):
                add_x(x)
                swapped = True
                break
        
        if not swapped:
            # Restore y
            add_x(y)
            
        # Optional: If we are stuck, we could try a random restart or larger kick
        # But simple swaps might be enough with randomization
        
    return S
