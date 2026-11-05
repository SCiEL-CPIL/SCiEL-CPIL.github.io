import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy set cover for other parameters
        total_words = q ** n
        
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx
            
        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return tuple(w)

        # Precompute neighbors
        neighbors = [None] * total_words
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.append(word_to_idx(tuple(nw)))
            neighbors[idx] = nbrs

        uncovered = [True] * total_words
        code = []
        
        while any(uncovered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                
        return [list(idx_to_word(idx)) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    
    # Precompute neighbors using numpy for speed
    # Each word covers itself + 16 neighbors = 17 words
    # Build a list of neighbor lists
    neighbors = []
    for idx in range(total_words):
        w = np.array([idx], dtype=np.int64)
        # Convert to base-3 digits
        digits = np.zeros(8, dtype=np.int64)
        temp = idx
        for i in range(7, -1, -1):
            digits[i] = temp % 3
            temp //= 3
        
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != digits[i]:
                    new_digits = digits.copy()
                    new_digits[i] = val
                    # Convert back to index
                    new_idx = 0
                    for v in new_digits:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors.append(nbrs)
    
    # Convert to numpy array of shape (total_words, 17) for faster operations
    neighbors_arr = np.array(neighbors, dtype=np.int64)
    
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    start_time = time.time()
    deadline = start_time + 115  # Leave 5s for cleanup and return
    
    # Strategy: Start with a good initial cover, then minimize using local search
    # 1. Build an initial cover using greedy with randomization
    
    # Use a product code as starting point: all combinations of first 4 and last 4 positions
    # This gives 3^4 * 3^4 = 729 words, but we can do better
    
    # Actually, let's use a smarter initial construction
    # Start with greedy set cover
    
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_list = []
    code_set = set()
    
    # Greedy construction with randomization
    rng = np.random.default_rng(42)
    
    while np.any(uncovered_mask) and time.time() < deadline - 25:
        # Compute coverage gain for each candidate
        # For speed, sample a subset of candidates
        remaining = [i for i in range(total_words) if i not in code_set]
        if len(remaining) > 500:
            candidates = rng.choice(remaining, size=500, replace=False)
        else:
            candidates = remaining
        
        best_idx = -1
        best_gain = -1
        
        for i in candidates:
            nbrs = neighbors_arr[i]
            gain = np.sum(uncovered_mask[nbrs])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        nbrs = neighbors_arr[best_idx]
        uncovered_mask[nbrs] = False

    # Ensure we have a valid cover
    if not np.any(uncovered_mask):
        pass
    else:
        # Complete the cover
        while np.any(uncovered_mask) and time.time() < deadline - 5:
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                nbrs = neighbors_arr[i]
                gain = np.sum(uncovered_mask[nbrs])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
                
            code_set.add(best_idx)
            code_list.append(best_idx)
            
            nbrs = neighbors_arr[best_idx]
            uncovered_mask[nbrs] = False

    # Now minimize using local search
    code_list = list(code_set)
    
    # Compute cover counts
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        nbrs = neighbors_arr[c]
        cover_count[nbrs] += 1

    def try_remove_redundant():
        removed = False
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                nbrs = neighbors_arr[c]
                if np.all(cover_count[nbrs] > 1):
                    code_list.pop(i)
                    cover_count[nbrs] -= 1
                    code_set.discard(c)
                    removed = True
                    changed = True
                    break
        return removed

    try_remove_redundant()

    # Local search: Kick and Prune with better heuristic
    max_swaps = 10000
    swaps_done = 0
    no_improve_count = 0
    best_size = len(code_list)
    
    while time.time() < deadline - 15 and swaps_done < max_swaps:
        # 1. Try to remove redundant words
        if try_remove_redundant():
            no_improve_count = 0
            if len(code_list) < best_size:
                best_size = len(code_list)
            continue
        
        # 2. No removal possible. Try a swap.
        if len(code_list) == 0:
            break
            
        # Pick a random word to remove
        idx_c = rng.integers(0, len(code_list))
        c = code_list[idx_c]
        
        # Temporarily remove c
        nbrs_c = neighbors_arr[c]
        cover_count[nbrs_c] -= 1
        
        # Find uncovered words (those with cover_count 0)
        uncovered_now = nbrs_c[cover_count[nbrs_c] == 0]
        
        if len(uncovered_now) == 0:
            # It was redundant
            code_list.pop(idx_c)
            code_set.discard(c)
            swaps_done += 1
            no_improve_count = 0
            continue
        
        # We need a replacement c' that covers all uncovered_now
        # c' must be in the intersection of neighbors[u] for u in uncovered_now
        
        # Calculate intersection using sets
        candidates = set(neighbors_arr[uncovered_now[0]].tolist())
        for u in uncovered_now[1:]:
            candidates.intersection_update(neighbors_arr[u].tolist())
            if not candidates:
                break
        
        # Remove current code words from candidates
        candidates.difference_update(code_set)
        
        if candidates:
            # Pick a candidate with best heuristic
            best_cp = -1
            best_score = -1
            
            cand_list = list(candidates)
            if len(cand_list) > 100:
                cand_list = rng.choice(cand_list, size=100, replace=False).tolist()
            
            for cp in cand_list:
                nbrs_cp = neighbors_arr[cp]
                # Score: maximize redundancy (words with cover_count >= 2)
                score = np.sum(cover_count[nbrs_cp] >= 2)
                if score > best_score:
                    best_score = score
                    best_cp = cp
            
            if best_cp != -1:
                # Perform swap
                code_list[idx_c] = best_cp
                code_set.discard(c)
                code_set.add(best_cp)
                
                nbrs_cp = neighbors_arr[best_cp]
                cover_count[nbrs_cp] += 1
                swaps_done += 1
                no_improve_count += 1
                
                # Try to remove redundant words after swap
                if try_remove_redundant():
                    no_improve_count = 0
                    if len(code_list) < best_size:
                        best_size = len(code_list)
            else:
                # Revert
                cover_count[nbrs_c] += 1
                swaps_done += 1
                no_improve_count += 1
        else:
            # No candidate found, revert
            cover_count[nbrs_c] += 1
            swaps_done += 1
            no_improve_count += 1
        
        # If no improvement for a while, try a more aggressive kick
        if no_improve_count > 1000:
            # Remove a random word and add a random replacement
            if len(code_list) > 500:
                idx_c = rng.integers(0, len(code_list))
                c = code_list[idx_c]
                nbrs_c = neighbors_arr[c]
                cover_count[nbrs_c] -= 1
                
                # Find a random word that covers the most uncovered
                uncovered_now = nbrs_c[cover_count[nbrs_c] == 0]
                if len(uncovered_now) > 0:
                    # Find candidates
                    candidates = set(neighbors_arr[uncovered_now[0]].tolist())
                    for u in uncovered_now[1:]:
                        candidates.intersection_update(neighbors_arr[u].tolist())
                        if not candidates:
                            break
                    candidates.difference_update(code_set)
                    
                    if candidates:
                        cp = rng.choice(list(candidates))
                        code_list[idx_c] = cp
                        code_set.discard(c)
                        code_set.add(cp)
                        cover_count[neighbors_arr[cp]] += 1
                        try_remove_redundant()
                    else:
                        cover_count[nbrs_c] += 1
                else:
                    code_list.pop(idx_c)
                    code_set.discard(c)
                    try_remove_redundant()
                
                no_improve_count = 0

    # Final cleanup
    try_remove_redundant()

    return [idx_to_word(c) for c in code_list]
