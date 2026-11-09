import random
import time
from collections import defaultdict

def _is_valid_add(x, chosen_set, counts):
    """Check if x can be added to chosen_set without violating B2[2]."""
    # x + y for y in chosen_set
    for y in chosen_set:
        s = x + y
        if counts[s] >= 2:
            return False
    # 2x
    if counts[2 * x] >= 2:
        return False
    return True

def _add_element(x, chosen_set, counts):
    """Add x to chosen_set and update counts."""
    for y in chosen_set:
        s = x + y
        counts[s] += 1
    counts[2 * x] += 1
    chosen_set.add(x)

def _remove_element(x, chosen_set, counts):
    """Remove x from chosen_set and update counts."""
    chosen_set.discard(x)
    for y in chosen_set:
        s = x + y
        counts[s] -= 1
    counts[2 * x] -= 1

def _random_greedy_init():
    """Initialize with a randomized greedy set."""
    chosen_set = set()
    counts = defaultdict(int)
    order = list(range(1, 1001))
    random.shuffle(order)
    for x in order:
        if _is_valid_add(x, chosen_set, counts):
            _add_element(x, chosen_set, counts)
    return chosen_set, counts

def construct():
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple Random Greedy Initializations
    # Find a good starting point
    while time.time() - start_time < timeout - 50:
        current_set, current_counts = _random_greedy_init()
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
    # Ensure we have a decent start
    if best_size == 0:
        current_set, current_counts = _random_greedy_init()
        best_set = current_set
        best_size = len(current_set)
        
    # Phase 2: Local Search
    current_set = best_set.copy()
    current_counts = defaultdict(int)
    # Rebuild counts for current_set
    for x in current_set:
        for y in current_set:
            if x <= y:
                current_counts[x+y] += 1
        # Note: The _add_element logic above adds counts for x+y for all y in set, including 2x.
        # My _remove_element and _add_element assume 'chosen' is a set.
        # Let's re-verify the count logic.
        # In _add_element(x, S, C):
        #   for y in S: C[x+y] += 1
        #   C[2x] += 1
        #   S.add(x)
        # This counts every pair (a,b) with a,b in S_new.
        # If S_old has size K, we add K pairs (x,y) where y in S_old, plus (x,x).
        # Total pairs added: K+1.
        # Correct.
        
        # My rebuild loop above was wrong. Let's just use the add function to rebuild or write a specific builder.
        # Easier: Just re-run the add logic on an empty set.
        current_counts = defaultdict(int)
        temp_set = set()
        for x in current_set:
            _add_element(x, temp_set, current_counts)
        current_set = temp_set # Just to be safe, though it's the same elements

    # Pre-generate a shuffled order for candidates to avoid re-shuffling 1000 items every loop
    # We will cycle through this list.
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Phase 2a: Try to add any element
        # We iterate through a random permutation of all numbers.
        # To avoid generating a new permutation every time (expensive), we can use a random step or just shuffle once and reuse?
        # Shuffling 1000 items is fast (~0.1ms). So shuffling every iteration is fine.
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        for x in candidates:
            if x not in current_set:
                if _is_valid_add(x, current_set, current_counts):
                    _add_element(x, current_set, current_counts)
                    improved = True
                    if len(current_set) > best_size:
                        best_set = current_set.copy()
                        best_size = len(current_set)
                    break # Restart search after addition to allow more additions
        
        if not improved:
            # Phase 2b: Try swaps
            # Remove a random element, try to add another.
            if current_set:
                u = random.choice(list(current_set))
                _remove_element(u, current_set, current_counts)
                
                # Try to find a valid x to add
                # We can bias towards numbers not in the set.
                # Simple approach: shuffle all candidates again.
                candidates = list(range(1, 1001))
                random.shuffle(candidates)
                
                swapped = False
                for x in candidates:
                    if x not in current_set:
                        if _is_valid_add(x, current_set, current_counts):
                            _add_element(x, current_set, current_counts)
                            swapped = True
                            # Size is same, so best_size doesn't update immediately, 
                            # but this new configuration might allow additions in next loop.
                            # We don't update best_set here because size didn't increase.
                            break
                
                if not swapped:
                    # Rollback
                    _add_element(u, current_set, current_counts)
                else:
                    improved = True
                    # Continue to next iteration to try adding more
                    # Note: We break out of the candidate loop, but not the while loop.
                    # The 'improved' flag ensures we don't break the while loop.
                    # We should probably try to add more immediately?
                    # The loop structure handles this: if improved, we go to next while iteration,
                    # which starts with Phase 2a (Additions).
        
        if not improved:
            # Stuck. Try a "kick" or restart.
            # If we are stuck, it means no single addition or single swap works.
            # Let's try a double swap: remove 2, add 2.
            # This is more expensive. Let's try a simpler restart strategy first.
            # Randomize the current set by removing a few random elements and trying to refill?
            # Or just do a new random greedy init if we've been stuck for a while?
            
            # To keep it simple and fast, let's just do a new random greedy init if we fail to improve for a few seconds?
            # Actually, let's try a "perturbation": remove 2 random elements, then try to add 2 random elements (greedily).
            if len(current_set) > 2:
                # Remove 2 random elements
                elems = list(current_set)
                random.shuffle(elems)
                u1, u2 = elems[0], elems[1]
                
                _remove_element(u1, current_set, current_counts)
                _remove_element(u2, current_set, current_counts)
                
                # Try to add 2 elements
                added_count = 0
                candidates = list(range(1, 1001))
                random.shuffle(candidates)
                
                for x in candidates:
                    if x not in current_set:
                        if _is_valid_add(x, current_set, current_counts):
                            _add_element(x, current_set, current_counts)
                            added_count += 1
                            if added_count == 2:
                                break
                
                if added_count < 2:
                    # Rollback if we couldn't refill
                    # This is complex to rollback perfectly if we added 1.
                    # Simpler: If we can't add 2, revert the whole move.
                    # But we already modified the set.
                    # Let's just accept the state if we added >= 1, or revert if 0?
                    # If we added 0, we have a smaller set. Bad.
                    # If we added 1, we have a smaller set. Bad.
                    # So we must revert if added_count < 2.
                    
                    # Revert: Remove what we added, re-add u1, u2.
                    # We didn't store what we added.
                    # This strategy is risky.
                    
                    # Alternative: Just do a full restart of the local search from a new random greedy init.
                    # This is expensive but safe.
                    # Let's limit restarts.
                    if time.time() - start_time < timeout - 10:
                        current_set, current_counts = _random_greedy_init()
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(current_set)
                        # Continue searching from this new state
                    else:
                        break
            else:
                break

    return list(best_set)
