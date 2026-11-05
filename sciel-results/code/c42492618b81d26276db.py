import time
import random
import sys

def _check_add(x, S, existing_sums):
    """
    Check if x can be added to S.
    Returns (is_valid, new_sums) where new_sums is the set of sums involving x.
    """
    n = len(S)
    new_sums = set()
    
    # 1. 3x
    s = 3 * x
    if s in existing_sums or s in new_sums:
        return False, None
    new_sums.add(s)
    
    # 2. 2x + y
    for y in S:
        s = 2 * x + y
        if s in existing_sums or s in new_sums:
            return False, None
        new_sums.add(s)
        
    # 3. x + 2y
    for y in S:
        s = x + 2 * y
        if s in existing_sums or s in new_sums:
            return False, None
        new_sums.add(s)
        
    # 4. x + y + z
    # We need to iterate y <= z in S
    # To be efficient, we can just iterate all pairs and add to set, 
    # but we must check against existing_sums and new_sums immediately to fail fast.
    for i in range(n):
        yi = S[i]
        for j in range(i, n):
            zj = S[j]
            s = x + yi + zj
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
    return True, new_sums

def _greedy_init():
    S = []
    existing_sums = set()
    for x in range(1, 4097):
        valid, new_sums = _check_add(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums.update(new_sums)
    return S, existing_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s safety margin
    
    # 1. Get a good initial solution
    S, existing_sums = _greedy_init()
    best_S = S[:]
    best_size = len(S)
    
    # Pre-compute a pool of candidates? 
    # Just use random sampling from 1..4096
    
    rng = random.Random(42)
    
    # Local Search
    # State: S, existing_sums
    # We will try to increase the size.
    # If we can't increase, we try to swap to find a configuration that allows increase.
    
    current_S = S[:]
    current_sums = existing_sums
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Try to add a random element
        # Sample a few candidates
        added = False
        for _ in range(50):
            x = rng.randint(1, 4096)
            if x in current_S:
                continue
            valid, new_sums = _check_add(x, current_S, current_sums)
            if valid:
                current_S.append(x)
                current_sums.update(new_sums)
                added = True
                if len(current_S) > best_size:
                    best_S = current_S[:]
                    best_size = len(current_S)
                no_improve_count = 0
                break
        
        if not added:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart or perturb
                # Perturb: remove a random element, then try to add
                if current_S:
                    idx = rng.randint(0, len(current_S)-1)
                    removed_x = current_S.pop(idx)
                    # Recalculate sums? 
                    # Removing sums is hard if we just have a set.
                    # We must rebuild sums.
                    # Rebuilding is O(k^3). k~30 -> 27000 ops. Acceptable.
                    current_sums = set()
                    # Rebuild sums for current_S
                    n = len(current_S)
                    for i in range(n):
                        for j in range(i, n):
                            for k in range(j, n):
                                current_sums.add(current_S[i] + current_S[j] + current_S[k])
                    
                    # Now try to add a few random elements to fill the gap
                    for _ in range(100):
                        x = rng.randint(1, 4096)
                        if x in current_S:
                            continue
                        valid, new_sums = _check_add(x, current_S, current_sums)
                        if valid:
                            current_S.append(x)
                            current_sums.update(new_sums)
                            if len(current_S) > best_size:
                                best_S = current_S[:]
                                best_size = len(current_S)
                            no_improve_count = 0
                            break
                else:
                    # If empty, restart greedy
                    current_S, current_sums = _greedy_init()
                    best_S = current_S[:]
                    best_size = len(current_S)
                    no_improve_count = 0

    return best_S
