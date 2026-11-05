import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    def is_valid(S):
        # Bitset DP
        reachable = 1  # bit 0 is set
        for x in S:
            reachable |= (reachable << x)
            # Early exit if 200 is reached
            if (reachable >> 200) & 1:
                return False
        return True

    def check_and_update(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            # We assume current_set is valid if it comes from valid moves, 
            # but for safety in initialization we might need to check.
            # In local search, we only accept valid moves.
            # In greedy, we only add if valid.
            # So we can skip is_valid here for speed if we are sure.
            # However, check_and_update is called from greedy where we know it's valid.
            # And from local search where we check before updating.
            best_set = list(current_set)
            best_size = len(current_set)
            
    # 1. Deterministic Greedies
    # Large to Small
    s = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # Small to Large
    s = []
    reachable = 1
    for x in range(1, 101):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # 2. Local search: start from a solution, try to add back excluded elements
    # Improved: Allow swaps to escape local optima
    def local_search(start_set, max_iters=20000):
        current = set(start_set)
        all_elems = set(range(1, 101))
        excluded = list(all_elems - current)
        best_local = list(current)
        best_local_size = len(current)
        
        # Pre-calculate mask for target 200 for faster check?
        # The is_valid function is the bottleneck if called too often.
        # We can optimize is_valid by passing the current reachable bitset?
        # No, because adding x changes reachable.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not excluded:
                break
            
            # Strategy: Randomly pick an excluded element to try to add.
            # If it fails, try to swap it with a random included element.
            
            x_idx = random.randint(0, len(excluded) - 1)
            x = excluded[x_idx]
            
            # Try Add
            new_set = current | {x}
            if is_valid(new_set):
                current = new_set
                excluded.pop(x_idx)
                if len(current) > best_local_size:
                    best_local = list(current)
                    best_local_size = len(current)
                    check_and_update(current)
            else:
                # Try Swap
                # Pick a random element y from current to remove
                if current:
                    y_idx = random.randint(0, len(current) - 1)
                    y = list(current)[y_idx] # O(N) to get list, might be slow.
                    # Better: keep current as a list? Or use set and random.choice?
                    # random.choice on set is O(N).
                    # Let's convert current to list once per iteration? No, that's expensive.
                    # Let's use a list for current in local search to allow fast indexing?
                    # But we need set operations for membership.
                    # Let's stick to set for membership, but for random selection, 
                    # we can just pick a random element from the set.
                    
                    y = random.choice(current)
                    new_set = (current - {y}) | {x}
                    if is_valid(new_set):
                        current = new_set
                        # Update excluded list: remove x, add y
                        excluded.pop(x_idx)
                        excluded.append(y)
                        # Size doesn't change, so no update to best_local_size
                        # But the structure changed, which might allow future additions.
                        # We don't update best_set here because size is same.
        
        return best_local
    
    # Run randomized greedy + local search
    for _ in range(100): # Increase restarts
        if time.time() > deadline:
            break
        
        order = list(range(1, 101))
        random.shuffle(order)
        s = []
        reachable = 1
        for x in order:
            new_reach = reachable | (reachable << x)
            if not ((new_reach >> 200) & 1):
                s.append(x)
                reachable = new_reach
        check_and_update(s)
        
        # Local search from this starting point
        improved = local_search(s, max_iters=5000)
        check_and_update(improved)
    
    # 3. Final polish: local search from best found
    if best_set:
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = local_search(best_set, max_iters=20000)
            check_and_update(improved)
    
    return best_set
