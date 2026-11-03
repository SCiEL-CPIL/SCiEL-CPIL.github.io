import itertools
import random
import time

def construct(n):
    # 1. Generate classic superpermutation (Initialization)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Stochastic Local Search / Iterated Local Search
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def get_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1
        return counts

    counts = get_counts(word)
    
    # Check if initial is valid
    if len(counts) != len(all_perms):
        return "".join(map(str, word))

    start_time = time.time()
    time_limit = 100.0 # Leave buffer for overhead
    
    def is_valid_deletion(w, idx, counts):
        # Check if deleting w[idx] keeps all perm counts >= 1
        j_min = max(0, idx - n + 1)
        j_max = min(idx, len(w) - n)
        
        lost = []
        for j in range(j_min, j_max + 1):
            p = tuple(w[j:j+n])
            if p in all_perms:
                if counts[p] == 1:
                    return False
                lost.append(p)
        
        # Check new windows formed
        # New word length is len(w)-1
        # Junction is between w[idx-1] and w[idx+1] (in original indices, now adjacent)
        # New windows in new_word covering the junction:
        # In new_word, the junction is at index idx-1 (between idx-1 and idx)
        # Windows starting at j' in [max(0, (idx-1)-n+1), min(idx-1, len(new_word)-n)]
        # Wait, let's just simulate the new windows' perms to be safe, 
        # but we only need to check if they *add* to counts. 
        # Since we only care if any count drops to 0, and we already checked 'lost' perms,
        # adding new perms only increases counts. 
        # So if no 'lost' perm had count 1, the deletion is valid.
        return True

    def try_hill_climb(w, counts):
        improved = True
        while improved:
            improved = False
            # Randomize order of indices to break symmetry? 
            # Or just iterate. Iteration is faster.
            # Let's iterate from 0 to len(w)-1
            for i in range(len(w)):
                if time.time() - start_time > time_limit:
                    return w, counts, True # timeout
                
                if is_valid_deletion(w, i, counts):
                    # Perform deletion
                    lost = []
                    j_min = max(0, i - n + 1)
                    j_max = min(i, len(w) - n)
                    for j in range(j_min, j_max + 1):
                        p = tuple(w[j:j+n])
                        if p in all_perms:
                            lost.append(p)
                            
                    for p in lost:
                        counts[p] -= 1
                    
                    # Update word
                    w = w[:i] + w[i+1:]
                    
                    # Update counts for new windows
                    j_min_new = max(0, i - n + 1)
                    j_max_new = min(i - 1, len(w) - n)
                    for j in range(j_min_new, j_max_new + 1):
                        p = tuple(w[j:j+n])
                        if p in all_perms:
                            counts[p] = counts.get(p, 0) + 1
                            
                    improved = True
                    break # Restart scan from 0? Or continue? 
                    # Restarting is safer for correctness of indices, but slower.
                    # Since we delete one char, indices shift. 
                    # If we continue, we might skip some. 
                    # Let's restart for robustness in hill climb phase.
                    # Actually, we can just break and restart the while loop.
        return w, counts, False

    def try_swap(w, i, j, counts):
        # Check if swapping w[i] and w[j] keeps validity
        # Affected windows are those containing i or j
        # Union of windows covering i and j
        affected_indices = set()
        
        for k in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
            affected_indices.add(k)
        for k in range(max(0, j - n + 1), min(j, len(w) - n) + 1):
            affected_indices.add(k)
            
        # Save old perms
        old_perms = {}
        for k in affected_indices:
            p = tuple(w[k:k+n])
            if p in all_perms:
                old_perms[k] = p
                
        # Perform swap in a copy or in-place? In-place is faster, but risky if invalid.
        # Let's simulate.
        w_copy = w[:]
        w_copy[i], w_copy[j] = w_copy[j], w_copy[i]
        
        # Check new perms
        valid = True
        new_perms = {}
        for k in affected_indices:
            p = tuple(w_copy[k:k+n])
            if p in all_perms:
                new_perms[k] = p
                # Check if this perm's count in 'counts' (which reflects 'w') 
                # minus the old ones plus new ones stays >= 1?
                # This logic is tricky with a global count dict.
                # Easier: 
                # 1. Decrement counts for old_perms
                # 2. Increment counts for new_perms
                # 3. Check if any count < 1
                # 4. If valid, keep swap. If not, revert counts.
                
        # Let's do the count check properly
        temp_counts = {}
        for p in old_perms.values():
             # We need to know how many times this perm appears in affected windows
             pass
            
        # Simpler approach for swap validity:
        # 1. Identify all windows affected.
        # 2. For each affected window, get old perm and new perm.
        # 3. Calculate delta for each permutation.
        # 4. If for any perm, count + delta < 1, invalid.
        
        deltas = {}
        for k in affected_indices:
            old_p = tuple(w[k:k+n])
            new_p = tuple(w_copy[k:k+n])
            
            if old_p in all_perms:
                deltas[old_p] = deltas.get(old_p, 0) - 1
            if new_p in all_perms:
                deltas[new_p] = deltas.get(new_p, 0) + 1
                
        for p, d in deltas.items():
            if counts.get(p, 0) + d < 1:
                valid = False
                break
                
        if valid:
            # Apply swap to w
            w[i], w[j] = w[j], w[i]
            # Update counts
            for p, d in deltas.items():
                counts[p] = counts.get(p, 0) + d
            return True
        return False

    best_word = word
    best_len = len(word)
    
    while time.time() - start_time < time_limit:
        # Phase 1: Hill Climbing (Deletions)
        word, counts, timed_out = try_hill_climb(word, counts)
        
        if len(word) < best_len:
            best_word = word[:]
            best_len = len(word)
            
        if timed_out:
            break
            
        # Phase 2: Perturbation (Swaps)
        # Try some random swaps to escape local optimum
        swaps_made = 0
        max_swaps = 20
        for _ in range(max_swaps):
            if time.time() - start_time > time_limit:
                break
            i = random.randint(0, len(word) - 1)
            j = random.randint(0, len(word) - 1)
            if i == j:
                continue
            if try_swap(word, i, j, counts):
                swaps_made += 1
                # After a swap, maybe we can delete again? 
                # The hill climb phase will run again in the next loop iteration.
                # But doing one deletion check immediately might be faster.
                # Let's just break out and let the main loop handle hill climbing.
                break 

    return "".join(map(str, best_word))
