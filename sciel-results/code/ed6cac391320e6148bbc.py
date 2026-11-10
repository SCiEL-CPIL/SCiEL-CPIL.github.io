import time
import random
import numpy as np

def _try_greedy_fast(order, deadline):
    """Fast greedy using numpy for vectorized checks."""
    chosen = []
    # counts[s] = number of representations of s
    # Use a dict for sparse counts
    counts = {}
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Generate new sums: x + y for y in chosen, and 2*x
        if chosen:
            new_sums = [x + y for y in chosen]
        else:
            new_sums = []
        new_sums.append(2 * x)
        
        # Check validity: no sum can have count >= 2 already
        valid = True
        for s in new_sums:
            if counts.get(s, 0) >= 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen

def _try_greedy_weighted(order, weights, deadline):
    """Greedy that prioritizes elements with higher 'desirability' scores."""
    # Sort by weight descending, then process
    indexed = list(enumerate(order))
    indexed.sort(key=lambda i: -weights[i[0]])
    
    chosen = []
    counts = {}
    
    for _, x in indexed:
        if time.time() > deadline:
            break
        
        if chosen:
            new_sums = [x + y for y in chosen]
        else:
            new_sums = []
        new_sums.append(2 * x)
        
        valid = True
        for s in new_sums:
            if counts.get(s, 0) >= 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen

def _local_search(initial_set, deadline, max_iter=1000):
    """Try to improve a set by swapping elements or adding/removing."""
    chosen = sorted(initial_set)
    chosen_set = set(chosen)
    
    # Build counts
    counts = {}
    for i, a in enumerate(chosen):
        for b in chosen[i:]:
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    
    # Verify validity
    for s, c in counts.items():
        if c > 2:
            return initial_set  # Invalid, return as-is
    
    best_size = len(chosen)
    best_set = list(chosen)
    
    all_elements = set(range(1, 1001))
    candidates_to_add = sorted(all_elements - chosen_set)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a candidate
        for x in candidates_to_add:
            if time.time() > deadline:
                break
            
            # Check if adding x is valid
            new_sums = [x + y for y in chosen] + [2 * x]
            valid = True
            for s in new_sums:
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                # Add x
                chosen.append(x)
                chosen_set.add(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
                best_size = len(chosen)
                best_set = list(chosen)
                improved = True
                break
        
        if improved:
            # Update candidates
            candidates_to_add = sorted(all_elements - chosen_set)
            continue
        
        # Try swapping: remove one, add another
        for i, x in enumerate(chosen):
            if time.time() > deadline:
                break
            
            # Temporarily remove x
            removed_sums = []
            for j, y in enumerate(chosen):
                if i == j:
                    s = 2 * x
                else:
                    s = x + y
                removed_sums.append(s)
            
            # Check if removal is safe (all sums have count >= 1)
            can_remove = True
            for s in removed_sums:
                if counts.get(s, 0) < 1:
                    can_remove = False
                    break
            
            if not can_remove:
                continue
            
            # Actually remove
            for s in removed_sums:
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            chosen.pop(i)
            chosen_set.discard(x)
            
            # Try adding each candidate
            for y in candidates_to_add:
                if time.time() > deadline:
                    break
                
                new_sums = [y + z for z in chosen] + [2 * y]
                valid = True
                for s in new_sums:
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    chosen.append(y)
                    chosen_set.add(y)
                    for s in new_sums:
                        counts[s] = counts.get(s, 0) + 1
                    best_size = len(chosen)
                    best_set = list(chosen)
                    improved = True
                    break
            
            if not improved:
                # Restore x
                chosen.insert(i, x)
                chosen_set.add(x)
                for s in removed_sums:
                    counts[s] = counts.get(s, 0) + 1
            
            if improved:
                candidates_to_add = sorted(all_elements - chosen_set)
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy (Ascending)
    s1 = _try_greedy_fast(range(1, 1001), deadline)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
    
    # 2. Deterministic Greedy (Descending)
    s2 = _try_greedy_fast(range(1000, 0, -1), deadline)
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
    
    # 3. Randomized Greedy
    rng = random.Random(42)
    
    # Run multiple randomized restarts
    while time.time() < deadline:
        if time.time() > deadline:
            break
        
        order = list(range(1, 1001))
        rng.shuffle(order)
        s = _try_greedy_fast(order, deadline)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        
        # Occasionally do local search on the best result
        if len(s) >= best_size and time.time() < deadline - 5.0:
            improved = _local_search(best_set, deadline, max_iter=500)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
    
    return best_set
