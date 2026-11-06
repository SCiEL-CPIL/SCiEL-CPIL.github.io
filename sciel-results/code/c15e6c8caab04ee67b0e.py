import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Generate initial greedy solution
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 not in new_sums:
            selected.append(x)
            reachable |= new_sums
    
    # Helper to check validity of a set given its reachable sums
    # We maintain the set and the reachable sums
    # For local search, we need to efficiently update reachable sums
    # Removing an element is hard with just a set of sums.
    # We might need to recompute reachable sums from scratch after a swap.
    # Recomputation is O(N * Sum) = 100 * 200 = 20000 ops. Very fast.
    
    def is_valid_and_get_reachable(S):
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 in new_sums:
                return False, None
            reachable |= new_sums
        return True, reachable

    current_set = set(selected)
    current_reachable, _ = is_valid_and_get_reachable(current_set)
    best_size = len(current_set)
    best_set = set(current_set)
    
    # Local Search
    # We want to maximize size.
    # Moves:
    # 1. Add x not in current_set
    # 2. Swap y in current_set with x not in current_set
    
    while time.time() < deadline:
        improved = False
        
        # Try to add
        for x in range(100, 0, -1):
            if x not in current_set:
                # Check if adding x is valid
                # New sums = current_reachable U {s+x for s in current_reachable}
                # Check if 200 in new sums
                conflict = False
                for s in current_reachable:
                    if s + x == 200:
                        conflict = True
                        break
                    if s + x > 200:
                        continue
                    # If s+x < 200, it's a new sum, doesn't cause conflict directly unless it's 200
                if not conflict:
                    # Also need to check if any existing sum s1 + x + ... ?
                    # No, if we add x, the new subset sums are old sums U {old_sum + x}.
                    # Conflict if 200 in {old_sum + x} => old_sum = 200 - x.
                    # So we just need to check if 200 - x is in current_reachable.
                    if (200 - x) not in current_reachable:
                        current_set.add(x)
                        _, current_reachable = is_valid_and_get_reachable(current_set)
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break # Restart add/swap loop
        
        if improved:
            continue
            
        # Try to swap
        # Randomize order to escape local optima
        candidates_to_remove = list(current_set)
        random.shuffle(candidates_to_remove)
        
        for y in candidates_to_remove:
            if time.time() > deadline:
                break
            candidates_to_add = [x for x in range(100, 0, -1) if x not in current_set]
            random.shuffle(candidates_to_add)
            
            for x in candidates_to_add:
                if time.time() > deadline:
                    break
                # Try swap y with x
                new_set = set(current_set)
                new_set.remove(y)
                new_set.add(x)
                
                valid, new_reachable = is_valid_and_get_reachable(new_set)
                if valid:
                    # Check if this swap allows adding more elements?
                    # For now, just accept the swap if it doesn't reduce size (it doesn't)
                    # But we want to increase size. A swap alone doesn't increase size.
                    # We only do a swap if it enables a subsequent addition?
                    # That's expensive.
                    # Alternative: Do the swap, then try to add.
                    # If add works, great. If not, revert?
                    # Let's try the swap, then try to add.
                    
                    current_set = new_set
                    current_reachable = new_reachable
                    
                    # Try to add immediately
                    added = False
                    for z in range(100, 0, -1):
                        if z not in current_set:
                            if (200 - z) not in current_reachable:
                                current_set.add(z)
                                _, current_reachable = is_valid_and_get_reachable(current_set)
                                added = True
                                if len(current_set) > best_size:
                                    best_size = len(current_set)
                                    best_set = set(current_set)
                                break
                    if added:
                        improved = True
                        break
                    else:
                        # Swap didn't lead to immediate improvement.
                        # Revert? Or keep?
                        # Keeping might lead to a better state later.
                        # But it risks cycling.
                        # Let's revert for now to keep it simple, or use a counter.
                        # Let's revert to avoid getting stuck in a plateau.
                        current_set.remove(x)
                        current_set.add(y)
                        _, current_reachable = is_valid_and_get_reachable(current_set)
            
            if improved:
                break
    
    return list(best_set)
