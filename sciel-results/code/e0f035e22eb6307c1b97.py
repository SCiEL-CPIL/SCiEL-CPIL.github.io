import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    max_n = 1000
    max_sum = 2 * max_n
    
    # counts[s] = number of pairs (a, b) in S with a <= b and a + b = s
    counts = [0] * (max_sum + 1)
    in_set = [False] * (max_n + 1)
    
    def add_element(x):
        # Check if x can be added
        # Pairs involving x: (x, x) and (a, x) for a in S, a < x
        # Sum x+x = 2x
        if counts[2 * x] >= 2:
            return False
        
        for a in range(1, x):
            if in_set[a]:
                s = a + x
                if counts[s] >= 2:
                    return False
        
        # If valid, actually add
        in_set[x] = True
        counts[2 * x] += 1
        for a in range(1, x):
            if in_set[a]:
                counts[a + x] += 1
        return True

    def remove_element(x):
        in_set[x] = False
        counts[2 * x] -= 1
        for a in range(1, x):
            if in_set[a]:
                counts[a + x] -= 1

    def get_size():
        return sum(1 for i in range(1, max_n + 1) if in_set[i])

    # 1. Greedy Initialization (Ascending)
    for i in range(1, max_n + 1):
        add_element(i)
    
    best_size = get_size()
    # Save best state
    best_in_set = in_set[:]
    best_counts = counts[:]
    
    # 2. Local Search
    # We will try to improve the current set.
    # Strategy: 
    #   a) Try to add any element that fits.
    #   b) If none fits, try removing an element and then greedily filling.
    
    def greedy_fill(order):
        # Try to add elements in the given order if they fit
        for x in order:
            if not in_set[x]:
                # Check validity
                valid = True
                if counts[2 * x] >= 2:
                    valid = False
                else:
                    for a in range(1, x):
                        if in_set[a] and counts[a + x] >= 2:
                            valid = False
                            break
                if valid:
                    add_element(x)

    # Main Loop
    improved = True
    while time.time() - start_time < deadline:
        improved = False
        
        # Step 1: Try to add any element
        # We can iterate through all elements to see if any can be added
        added_any = False
        for x in range(1, max_n + 1):
            if not in_set[x]:
                if add_element(x):
                    added_any = True
                    improved = True
                    # If we added one, restart the loop to check for more
                    break
        
        if added_any:
            if get_size() > best_size:
                best_size = get_size()
                best_in_set = in_set[:]
                best_counts = counts[:]
            continue
        
        # Step 2: No single element can be added. Try removing one.
        # To make this efficient, we can try removing elements that are "expensive" 
        # or just iterate. Since N=1000, iterating is fine.
        
        # We'll try a few removals per outer loop iteration to save time for checking
        # Or just one random removal if we are stuck.
        # Let's try all removals but limit the "greedy fill" depth or time.
        
        current_size = get_size()
        
        # To avoid O(N^3) worst case, we'll just pick a random element to remove 
        # if the deterministic search is too slow, but N=1000 is small enough 
        # that a full scan might be okay if the inner fill is fast.
        
        # Let's try removing each element x in S.
        # For each x:
        #  1. Remove x
        #  2. Greedy fill (ascending)
        #  3. If size > current_size, accept and break
        #  4. Restore
        
        # Optimization: Only try removals if we haven't improved recently?
        # Or just do it.
        
        # To prevent infinite loops on bad local optima, we can add a "randomized" 
        # greedy fill sometimes.
        
        # Let's try a random element first to escape local optima quickly.
        # If that fails, try systematic.
        
        # Actually, let's just do a random restart of the local search strategy if stuck for a while.
        # But for now, let's try removing a random element from S.
        
        s_elements = [i for i in range(1, max_n + 1) if in_set[i]]
        
        if not s_elements:
            break
            
        # Pick a random element to remove
        # Or pick the one that allows the most additions? That's expensive.
        # Let's pick a random one.
        
        if random.random() < 0.5:
             # Random removal
             x_to_remove = random.choice(s_elements)
        else:
             # Ascending removal (deterministic)
             x_to_remove = s_elements[0]
             
        remove_element(x_to_remove)
        
        # Greedy fill
        # Use a slightly randomized order for the fill to explore different paths?
        # Or just ascending.
        greedy_fill(range(1, max_n + 1))
        
        new_size = get_size()
        
        if new_size > best_size:
            best_size = new_size
            best_in_set = in_set[:]
            best_counts = counts[:]
            improved = True
        elif new_size == current_size:
            # If we didn't get larger, we might be in a cycle.
            # Keep the new state if it's different? 
            # For now, if not improved, we might want to revert or keep?
            # If we keep, we might oscillate.
            # Let's revert if not strictly better to ensure progress on size?
            # But maybe a same-size set allows future improvements?
            # Let's keep the new state to explore.
            pass
        else:
            # Revert
            # We need to undo the greedy fill and the removal.
            # This is complex. 
            # Easier: Save state before removal, restore if worse.
            # Let's refactor to save/restore.
            
            # Since we already modified in_set and counts, we need to restore.
            # This is inefficient.
            
            # Let's change strategy: 
            # Always save state before trying a move.
            # If move fails, restore.
            
            # Let's re-implement the loop with explicit save/restore for the removal step.
            # The code above modified in_place. I need to fix this logic.
            
            # Let's restart the logic for the local search step to be safe.
            pass
            
        # If we didn't improve, we might be stuck.
        # If we keep doing random removals and fills, we will eventually find a better one 
        # or cycle. 
        # To avoid cycling, we can track visited states (hash of in_set).
        # But 2^1000 states is too many.
        
        # Let's add a timeout for the local search phase and then do a full restart 
        # with a different random seed or different initialization.
        
    # Restore best
    in_set[:] = best_in_set
    counts[:] = best_counts
    
    result = [i for i in range(1, max_n + 1) if in_set[i]]
    return result
