import math
import time

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    n = 255
    # Known valid solution: multiples of 3 (size 85)
    best_set = list(range(3, n + 1, 3))
    best_size = len(best_set)
    
    # Build candidate set: numbers in [1,255] sharing a factor with 255
    candidates = []
    for i in range(1, n + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    if 255 not in candidates:
        candidates.append(255)
    
    num_cands = len(candidates)
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = cand_to_idx[255]
    
    # Build adjacency: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * num_cands for _ in range(num_cands)]
    for i in range(num_cands):
        for j in range(i + 1, num_cands):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Try to find a clique larger than 85 using branch and bound
    # Start with the multiples-of-3 clique as initial best
    # Use a greedy coloring bound for branch and bound
    
    # Precompute color classes for upper bound
    # Simpler approach: try adding one element at a time to the base clique
    
    # First, identify which candidates can potentially be added
    # A candidate x can be in a clique with all multiples of 3 iff 
    # gcd(x, m) > 1 for all multiples m of 3 in [1,255]
    # This is very restrictive. Let's check each non-multiple-of-3 candidate.
    
    multiples_of_3 = set(range(3, n + 1, 3))
    
    # For each candidate not in multiples_of_3, check if it's compatible with ALL multiples of 3
    addable = []
    for x in candidates:
        if x in multiples_of_3:
            continue
        # Check if x is compatible with all multiples of 3
        compatible_with_all = True
        for m in multiples_of_3:
            if math.gcd(x, m) == 1:
                compatible_with_all = False
                break
        if compatible_with_all:
            addable.append(x)
    
    # If we can add any of these, we'd have > 85
    # But they also need to be pairwise compatible with each other
    if addable and time.time() < deadline:
        # Check pairwise compatibility among addable
        valid_addable = []
        for i, x in enumerate(addable):
            ok = True
            for j in range(i + 1, len(addable)):
                if math.gcd(x, addable[j]) == 1:
                    ok = False
                    break
            if ok:
                valid_addable.append(x)
        
        # Try adding all valid_addable to the base
        trial_set = set(best_set)
        for x in valid_addable:
            # Verify x is compatible with all in trial_set
            can_add = True
            for y in trial_set:
                if math.gcd(x, y) == 1:
                    can_add = False
                    break
            if can_add:
                trial_set.add(x)
        
        if len(trial_set) > best_size:
            best_set = list(trial_set)
            best_size = len(best_set)
    
    # More aggressive: try a different base clique
    # Try multiples of 5 as base
    multiples_of_5 = set(range(5, n + 1, 5))
    if len(multiples_of_5) < best_size:
        # Try to expand multiples of 5
        addable5 = []
        for x in candidates:
            if x in multiples_of_5:
                continue
            compatible_with_all = True
            for m in multiples_of_5:
                if math.gcd(x, m) == 1:
                    compatible_with_all = False
                    break
            if compatible_with_all:
                addable5.append(x)
        
        if addable5:
            trial5 = set(multiples_of_5)
            for x in addable5:
                can_add = True
                for y in trial5:
                    if math.gcd(x, y) == 1:
                        can_add = False
                        break
                if can_add:
                    trial5.add(x)
            if len(trial5) > best_size:
                best_set = list(trial5)
                best_size = len(best_set)
    
    # Try a local search / hill climbing approach
    # Start from best_set, try swapping elements
    if time.time() < deadline - 5:
        current_set = set(best_set)
        current_size = len(current_set)
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and time.time() < deadline - 5 and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try removing one element and adding two
            for x in list(current_set):
                if time.time() > deadline - 5:
                    break
                temp_set = current_set - {x}
                temp_size = len(temp_set)
                
                # Try adding candidates not in temp_set
                added = 0
                for y in candidates:
                    if y in temp_set:
                        continue
                    if time.time() > deadline - 5:
                        break
                    # Check if y is compatible with all in temp_set
                    can_add = True
                    for z in temp_set:
                        if math.gcd(y, z) == 1:
                            can_add = False
                            break
                    if can_add:
                        temp_set.add(y)
                        added += 1
                        if added >= 2:
                            break
                
                if len(temp_set) > current_size:
                    current_set = temp_set
                    current_size = len(current_set)
                    improved = True
                    if current_size > best_size:
                        best_set = list(current_set)
                        best_size = current_size
                    break  # Restart search from new state
        
        if current_size > best_size:
            best_set = list(current_set)
            best_size = current_size
    
    # Final verification
    if time.time() < deadline - 2:
        is_valid = True
        bs = set(best_set)
        if 255 not in bs:
            is_valid = False
        else:
            for i, a in enumerate(best_set):
                for b in best_set[i+1:]:
                    if math.gcd(a, b) == 1:
                        is_valid = False
                        break
                if not is_valid:
                    break
        
        if not is_valid:
            # Fallback to known valid solution
            best_set = list(range(3, n + 1, 3))
    
    return best_set
