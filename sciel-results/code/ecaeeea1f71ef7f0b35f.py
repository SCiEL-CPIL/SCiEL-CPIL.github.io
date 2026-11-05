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
    
    # Precompute neighbors for all words
    # A word w has neighbors: w itself + all words at distance 1
    # For q=3, n=8, radius=1, each word covers 1 + 8*2 = 17 words
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    # Convert back to index
                    new_idx = 0
                    for v in nw:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors[idx] = nbrs

    # Convert to numpy arrays for faster operations
    # neighbors is a list of lists, but we can work with it directly
    
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    start_time = time.time()
    deadline = start_time + 110  # Leave 10s for cleanup and return

    # Strategy: Start with a random valid cover, then minimize using local search
    # 1. Build an initial cover using greedy
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_list = []
    code_set = set()
    
    # Greedy construction
    while np.any(uncovered_mask) and time.time() < deadline - 20:
        best_idx = -1
        best_gain = -1
        # Sample a subset of candidates to speed up
        candidates = [i for i in range(total_words) if i not in code_set]
        if len(candidates) > 200:
            candidates = random.sample(candidates, 200)
        
        for i in candidates:
            gain = 0
            for nbr in neighbors[i]:
                if uncovered_mask[nbr]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            uncovered_mask[nbr] = False

    if not np.any(uncovered_mask):
        # We have a valid cover. Now minimize it.
        pass
    else:
        # If greedy didn't finish, we need to complete it.
        # This is a fallback, but greedy should usually finish.
        while np.any(uncovered_mask) and time.time() < deadline - 5:
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                gain = 0
                for nbr in neighbors[i]:
                    if uncovered_mask[nbr]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
                
            code_set.add(best_idx)
            code_list.append(best_idx)
            
            for nbr in neighbors[best_idx]:
                uncovered_mask[nbr] = False

    # Now we have a valid cover in code_list.
    # Minimize using local search:
    # - Try to remove a codeword if it's redundant (all its neighbors are covered by others).
    # - Try to swap a codeword with another if it allows removing more later.
    
    code_list = list(code_set)
    
    # Compute cover counts
    cover_count = np.zeros(total_words, dtype=int)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    def try_remove_redundant():
        removed = False
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                redundant = True
                for nbr in neighbors[c]:
                    if cover_count[nbr] <= 1:
                        redundant = False
                        break
                if redundant:
                    code_list.pop(i)
                    for nbr in neighbors[c]:
                        cover_count[nbr] -= 1
                    removed = True
                    changed = True
                    break
        return removed

    try_remove_redundant()

    # Local search: Kick and Prune with improved heuristics
    
    max_swaps = 8000
    swaps_done = 0
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 15 and swaps_done < max_swaps and no_improve_count < max_no_improve:
        # 1. Try to remove redundant words (multi-pass)
        removed_any = try_remove_redundant()
        if removed_any:
            no_improve_count = 0
            continue
        
        # 2. No removal possible. Try a swap.
        if len(code_list) == 0:
            break
        
        # Pick a random word to remove
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]
        
        # Temporarily remove c
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1
            
        # Find uncovered words (those with cover_count 0)
        uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
        
        if not uncovered_now:
            # It was redundant, but try_remove_redundant missed it?
            # Just remove it.
            code_list.pop(idx_c)
            code_set.discard(c)
            no_improve_count = 0
            continue
        
        # We need a replacement c' that covers all uncovered_now.
        # c' must be in the intersection of neighbors[u] for u in uncovered_now.
        
        # Calculate intersection
        candidates = set(neighbors[uncovered_now[0]])
        for u in uncovered_now[1:]:
            candidates.intersection_update(neighbors[u])
            if not candidates:
                break
        
        # Remove current code words from candidates (we want a NEW word)
        candidates.difference_update(code_set)
        
        if candidates:
            # Pick a candidate with improved scoring:
            # Score = (number of neighbors with cover_count >= 2) * 2 
            #        - (number of neighbors with cover_count == 1)
            # This favors words that create more redundancy (enabling future removals)
            # while penalizing words that don't help with current tight spots
            
            best_cp = -1
            best_score = -10**9
            
            cand_list = list(candidates)
            if len(cand_list) > 80:
                cand_list = random.sample(cand_list, 80)
            
            for cp in cand_list:
                score = 0
                for nbr in neighbors[cp]:
                    cc = cover_count[nbr]
                    if cc >= 2:
                        score += 2
                    elif cc == 1:
                        score -= 1
                if score > best_score:
                    best_score = score
                    best_cp = cp
            
            if best_cp != -1:
                # Perform swap
                code_list[idx_c] = best_cp
                code_set.discard(c)
                code_set.add(best_cp)
                
                for nbr in neighbors[best_cp]:
                    cover_count[nbr] += 1
                
                # After swap, try immediate removals
                if try_remove_redundant():
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                no_improve_count += 1
        else:
            # No candidate found, revert
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
            no_improve_count += 1

    # Final aggressive polish: multiple rounds of removal + targeted swaps
    for _ in range(3):
        if time.time() >= deadline - 5:
            break
        try_remove_redundant()
        
        # One more round of swaps focused on high-removal-potential words
        for _ in range(50):
            if time.time() >= deadline - 5:
                break
            if not code_list:
                break
            
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered_now:
                code_list.pop(idx_c)
                code_set.discard(c)
                try_remove_redundant()
                continue
            
            candidates = set(neighbors[uncovered_now[0]])
            for u in uncovered_now[1:]:
                candidates.intersection_update(neighbors[u])
                if not candidates:
                    break
            
            candidates.difference_update(code_set)
            
            if candidates:
                best_cp = -1
                best_score = -10**9
                cand_list = list(candidates)
                if len(cand_list) > 40:
                    cand_list = random.sample(cand_list, 40)
                
                for cp in cand_list:
                    score = 0
                    for nbr in neighbors[cp]:
                        cc = cover_count[nbr]
                        if cc >= 2:
                            score += 2
                        elif cc == 1:
                            score -= 1
                    if score > best_score:
                        best_score = score
                        best_cp = cp
                
                if best_cp != -1:
                    code_list[idx_c] = best_cp
                    code_set.discard(c)
                    code_set.add(best_cp)
                    for nbr in neighbors[best_cp]:
                        cover_count[nbr] += 1
                    try_remove_redundant()
                else:
                    for nbr in neighbors[c]:
                        cover_count[nbr] += 1
            else:
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1

    # Final cleanup
    try_remove_redundant()

    return [idx_to_word(c) for c in code_list]
