import random
import time
import math

def construct(q, n, radius):
    total_words = q ** n
    
    # Helper to convert index to tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    all_words = [idx_to_tuple(i) for i in range(total_words)]
    
    # Precompute neighbors as lists of indices
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        neighbors[i].append(i)
        w = all_words[i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = (new_val - orig_val) * (q ** (n - 1 - pos))
                neighbors[i].append(i + diff)
                
    # Greedy Initialization with multiple restarts
    def greedy_init():
        covered = [False] * total_words
        uncovered_count = total_words
        centers = []
        # Randomize order of candidates to diversify
        order = list(range(total_words))
        random.shuffle(order)
        
        while uncovered_count > 0:
            best_idx = -1
            best_gain = -1
            # Sample a subset of candidates to speed up? 
            # For N=6561, full scan is ~6561 * 17 * 6561/2? No.
            # Each step scans 6561 candidates. 6561 steps.
            # 6561 * 6561 * 17 ops is too slow in Python? 
            # 43M * 17 = 700M ops. Too slow.
            
            # Optimization: Only check candidates that cover at least 1 uncovered word.
            # Or use a heuristic: pick random uncovered word, pick neighbor that covers most uncovered.
            
            # Let's use the "random uncovered word" heuristic.
            # 1. Pick a random uncovered word u.
            # 2. Look at all neighbors of u (potential centers).
            # 3. Pick the one that covers the most *other* uncovered words.
            
            # To pick random uncovered word, maintain a list? 
            # Or just pick random index until uncovered?
            
            # For simplicity and speed, let's stick to a slightly optimized greedy:
            # Iterate through a random permutation of all words.
            # But we need to pick the *best* center for the current uncovered set.
            
            # Alternative: 
            # Maintain a list of uncovered words.
            # While list not empty:
            #   Pick random u from list.
            #   Find c in neighbors[u] that maximizes gain.
            #   Add c.
            #   Remove all covered words from list.
            
            uncovered_list = list(range(total_words))
            random.shuffle(uncovered_list)
            
            while uncovered_list:
                # Pick the first one (randomly ordered)
                u = uncovered_list[0]
                
                best_c = -1
                best_gain = -1
                
                # Candidates are neighbors of u
                for c in neighbors[u]:
                    # Calculate gain: how many words in uncovered_list are in neighbors[c]?
                    # This is expensive if we check all.
                    # Approximation: Just pick the c that covers u and maybe some others.
                    # Let's just count for a few random neighbors?
                    # No, let's do a proper scan of neighbors[u] (size 17).
                    # For each c, count how many of its neighbors are in uncovered.
                    # We need a fast way to check "is x in uncovered".
                    # Use a set for uncovered?
                    
                    # Rebuilding set every step is expensive.
                    # Let's use an array `is_uncovered` and just iterate neighbors[c].
                    gain = 0
                    for nb in neighbors[c]:
                        if is_uncovered[nb]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                        
                # Add best_c
                centers.append(best_c)
                
                # Update is_uncovered and uncovered_list
                # This is O(N) to update list, but we can just filter?
                # Or mark and filter at the end?
                # Since N=6561, filtering is fine.
                
                for nb in neighbors[best_c]:
                    is_uncovered[nb] = False
                    
                # Rebuild uncovered_list? 
                # Or just remove from list?
                # Removing from list is O(N).
                # Let's just create a new list? 
                # No, let's use the array and just pick random indices that are uncovered.
                
                # Actually, the "random uncovered word" strategy with full gain calculation 
                # for 17 candidates is fast.
                # The bottleneck is updating the uncovered state.
                
                # Let's simplify: 
                # We don't need to maintain `uncovered_list` perfectly if we just pick random indices 
                # and check `is_uncovered`.
                
                pass # The logic above is a bit mixed. Let's implement a clean version.
                
        return centers

    # Clean Greedy
    def greedy_init_clean():
        is_uncovered = [True] * total_words
        uncovered_count = total_words
        centers = []
        
        # Pre-compute a random order of all words to break ties/diversify
        # Actually, the "pick random uncovered, find best neighbor" is good.
        
        while uncovered_count > 0:
            # Find a random uncovered word
            # To avoid O(N) search every time, we can just pick random indices 
            # and check, but expected time is small if we are near end? 
            # No, at start, probability is high.
            
            u = -1
            while u == -1 or not is_uncovered[u]:
                u = random.randint(0, total_words - 1)
                
            # Find best center covering u
            best_c = -1
            best_gain = -1
            
            for c in neighbors[u]:
                gain = 0
                for nb in neighbors[c]:
                    if is_uncovered[nb]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
            centers.append(best_c)
            
            for nb in neighbors[best_c]:
                if is_uncovered[nb]:
                    is_uncovered[nb] = False
                    uncovered_count -= 1
                    
        return centers

    # Run multiple greedy restarts
    best_centers = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit_init = 10.0
    
    while time.time() - start_time < time_limit_init:
        c = greedy_init_clean()
        if len(c) < best_len:
            best_len = len(c)
            best_centers = c
            
    # Local Search: Simulated Annealing
    # State: centers (list), cover_count (array)
    
    centers = best_centers
    cover_count = [0] * total_words
    for c in centers:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    def get_uncovered_count():
        count = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                count += 1
        return count

    # We want to minimize len(centers) subject to uncovered_count == 0.
    # Energy = len(centers) + L * uncovered_count
    
    L = 100.0 # High penalty for uncovered
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    current_energy = len(centers) + L * get_uncovered_count()
    
    start_time_ls = time.time()
    max_time_ls = 105.0
    
    while time.time() - start_time_ls < max_time_ls and T > T_min:
        # Pick a move
        move_type = random.random()
        
        old_size = len(centers)
        old_unc = get_uncovered_count()
        old_energy = old_size + L * old_unc
        
        new_centers = centers.copy() # Expensive!
        
        # Optimization: Don't copy. Apply move, check, revert if rejected.
        
        if move_type < 0.3:
            # Add a random center
            c_new = random.randint(0, total_words - 1)
            if c_new not in centers:
                # Apply
                centers.append(c_new)
                for nb in neighbors[c_new]:
                    cover_count[nb] += 1
                
                new_size = len(centers)
                new_unc = get_uncovered_count()
                new_energy = new_size + L * new_unc
                
                delta = new_energy - old_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    pass # Accept
                else:
                    # Revert
                    centers.pop()
                    for nb in neighbors[c_new]:
                        cover_count[nb] -= 1
                    
        elif move_type < 0.6:
            # Remove a random center
            if centers:
                idx = random.randint(0, len(centers) - 1)
                c_old = centers[idx]
                
                # Apply
                centers.pop(idx)
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                    
                new_size = len(centers)
                new_unc = get_uncovered_count()
                new_energy = new_size + L * new_unc
                
                delta = new_energy - old_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    pass # Accept
                else:
                    # Revert
                    centers.insert(idx, c_old)
                    for nb in neighbors[c_old]:
                        cover_count[nb] += 1
                    
        else:
            # Swap: Remove c_old, Add c_new
            if centers:
                idx = random.randint(0, len(centers) - 1)
                c_old = centers[idx]
                c_new = random.randint(0, total_words - 1)
                
                if c_new != c_old:
                    # Apply
                    centers[idx] = c_new
                    for nb in neighbors[c_old]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c_new]:
                        cover_count[nb] += 1
                        
                    new_size = len(centers)
                    new_unc = get_uncovered_count()
                    new_energy = new_size + L * new_unc
                    
                    delta = new_energy - old_energy
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        pass # Accept
                    else:
                        # Revert
                        centers[idx] = c_old
                        for nb in neighbors[c_new]:
                            cover_count[nb] -= 1
                        for nb in neighbors[c_old]:
                            cover_count[nb] += 1
        
        T *= cooling
        
    # Final Repair: If not valid, greedily add centers
    if get_uncovered_count() > 0:
        is_uncovered = [cover_count[i] == 0 for i in range(total_words)]
        while any(is_uncovered):
            u = -1
            while not is_uncovered[u]:
                u = random.randint(0, total_words - 1)
            best_c = -1
            best_gain = -1
            for c in neighbors[u]:
                gain = sum(1 for nb in neighbors[c] if is_uncovered[nb])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            centers.append(best_c)
            for nb in neighbors[best_c]:
                if is_uncovered[nb]:
                    is_uncovered[nb] = False
                    cover_count[nb] += 1 # Update cover_count too? 
                    # Actually, we should rebuild cover_count at the end to be safe.

    # Rebuild cover_count for final check/cleanup
    cover_count = [0] * total_words
    for c in centers:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    # Final Reduction Pass
    changed = True
    while changed:
        changed = False
        for i in range(len(centers)):
            c = centers[i]
            redundant = True
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    redundant = False
                    break
            if redundant:
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                centers.pop(i)
                changed = True
                break

    return [list(all_words[c]) for c in centers]
