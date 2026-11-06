import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # State
    S = []
    counts = {}
    
    def try_add(x):
        nonlocal S, counts
        # Check sums
        # 2*x
        if counts.get(2*x, 0) >= 2:
            return False
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        # Valid
        S.append(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in S[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        return True

    def remove(x):
        nonlocal S, counts
        S.remove(x)
        counts[2*x] -= 1
        for y in S:
            s = x + y
            counts[s] -= 1

    # Initialize with greedy
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    for x in candidates:
        try_add(x)
        
    best_size = len(S)
    
    # Local Search
    while time.time() < deadline:
        # Try to add
        added = False
        # Get candidates not in S
        # Using a set for S is faster for lookup
        S_set = set(S)
        candidates = [x for x in range(1, 1001) if x not in S_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if try_add(x):
                added = True
                best_size = len(S)
                break
        
        if not added:
            # Try swap: remove 1, add 2
            swapped = False
            # Try a few random removals
            for _ in range(50):
                if time.time() > deadline: break
                x_remove = random.choice(S)
                remove(x_remove)
                
                # Try to add 2
                added_count = 0
                S_set = set(S)
                candidates = [x for x in range(1, 1001) if x not in S_set]
                random.shuffle(candidates)
                
                for y in candidates:
                    if added_count >= 2: break
                    if try_add(y):
                        added_count += 1
                
                if added_count == 2:
                    best_size = len(S)
                    swapped = True
                    break
                else:
                    # Revert if we didn't add 2? 
                    # If we added 1, we are at same size. 
                    # If we added 0, we are at size-1.
                    # If we added 1, we should probably keep it to explore?
                    # But we want to maximize size. 
                    # If we added 1, we haven't improved. 
                    # Let's revert to be safe and try another removal.
                    # Or keep it? Keeping it might lead to a better state later.
                    # Let's keep it if added_count >= 1?
                    # Actually, if we added 1, we are at original size.
                    # If we added 0, we are at size-1.
                    # Let's revert if added_count < 2 to maintain size?
                    # No, if we added 1, we changed the set. 
                    # Let's just revert if added_count < 2 to ensure we don't get stuck in a smaller set.
                    # Wait, if we added 1, we can try to add another?
                    # The loop above tried to add 2. If it failed on the 2nd, it stopped.
                    # So we have added 1.
                    # Let's keep the state with 1 added? 
                    # It's a random walk.
                    # But we want to increase size.
                    # Let's revert if we didn't increase size.
                    while len(S) > best_size - 1: # This is messy
                         pass
                    
                    # Simpler: if added_count < 2, revert the additions?
                    # We don't have a stack.
                    # Let's just revert the removal? No, we already removed.
                    # Let's just accept the state if size >= best_size?
                    # If added_count == 1, size is same.
                    # If added_count == 0, size is -1.
                    # Let's revert if size < best_size.
                    # We can't easily revert.
                    # Let's just continue. The next iteration will try to add.
                    # If we are at size-1, we will try to add.
                    # So it's fine.
                    pass
            
            if not swapped:
                # No improvement in this batch
                pass

    return S
