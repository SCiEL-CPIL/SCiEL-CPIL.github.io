import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy set cover
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
                if idx in code:
                    continue
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
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx
        
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return tuple(w)

    # Precompute neighbors for all words
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nbrs.append(word_to_idx(tuple(nw)))
        neighbors[idx] = nbrs

    # Convert to sets for faster intersection operations later
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    # 1. Greedy Construction
    uncovered = np.ones(total_words, dtype=bool)
    code_list = []
    code_set = set()
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s for cleanup and return
    
    # Simple greedy: pick word covering most uncovered
    while np.any(uncovered) and time.time() < deadline:
        best_idx = -1
        best_gain = -1
        # Optimization: We can sample or just scan. Scanning 6561 is fast.
        for i in range(total_words):
            if i in code_set:
                continue
            gain = 0
            for nbr in neighbors[i]:
                if uncovered[nbr]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False

    # 2. Local Search / Pruning
    # Convert to list for easy manipulation
    code_list = list(code_set)
    
    # Compute cover counts
    cover_count = np.zeros(total_words, dtype=int)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    # Phase 2a: Remove redundant words
    # A word c is redundant if for all nbr in neighbors[c], cover_count[nbr] > 1
    # We iterate until no more removals.
    
    def try_remove_redundant():
        removed = False
        changed = True
        while changed and time.time() < deadline:
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
                    break # Restart scan to maintain consistency
        return removed

    try_remove_redundant()

    # Phase 2b: Kick and Prune
    # Try to swap a word with another to enable further removals.
    # We accept a swap if it leads to a net reduction in code size after pruning.
    # To speed this up, we limit the number of swap attempts.
    
    swap_attempts = 0
    max_swap_attempts = 20000 # Tune based on time
    
    # Precompute "potential" for each word? 
    # For a candidate replacement c', we want it to cover words that are currently covered exactly once 
    # (creating redundancy) or words that are uncovered (which shouldn't happen if we start from a valid code).
    # Actually, when we remove c, some words become uncovered (count 0). 
    # We need c' to cover ALL words that became uncovered.
    
    while time.time() < deadline and swap_attempts < max_swap_attempts:
        # 1. Try to remove
        if try_remove_redundant():
            continue # Restart swap search after successful removal
        
        # 2. No removal possible. Try a swap.
        if len(code_list) == 0:
            break
            
        # Pick a random word to remove
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]
        
        # Temporarily remove c
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1
            
        # Find uncovered words
        uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
        
        if not uncovered_now:
            # Should have been removable, but maybe my redundancy check missed it?
            # Or cover_count logic error. Just remove it.
            code_list.pop(idx_c)
            continue
        
        # Find a replacement c' that covers all uncovered_now
        # c' must be in the intersection of neighbors[u] for u in uncovered_now.
        
        # Calculate intersection of neighbor sets of uncovered words
        # Start with set of neighbors of first uncovered word
        candidates = set(neighbors[uncovered_now[0]])
        for u in uncovered_now[1:]:
            candidates.intersection_update(neighbor_sets[u])
            if not candidates:
                break
        
        # Remove current code words from candidates
        candidates.difference_update(code_list)
        
        if candidates:
            # Pick a candidate. 
            # Heuristic: Pick the one that maximizes the minimum cover count of its neighbors?
            # Or just pick random?
            # Let's pick the one that adds the most redundancy to existing covered words.
            # Score(c') = sum(cover_count[nbr] for nbr in neighbors[c'])
            # This might be expensive. 
            # Alternative: Just pick random from candidates.
            
            best_cp = -1
            best_score = -1
            # Limit candidate evaluation if too many
            cand_list = list(candidates)
            if len(cand_list) > 50:
                cand_list = random.sample(cand_list, 50)
            
            for cp in cand_list:
                # Score: how many neighbors of cp have cover_count >= 2?
                # This indicates it's adding redundancy.
                score = 0
                for nbr in neighbors[cp]:
                    if cover_count[nbr] >= 2:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cp = cp
            
            if best_cp != -1:
                # Perform swap
                code_list[idx_c] = best_cp
                for nbr in neighbors[best_cp]:
                    cover_count[nbr] += 1
                swap_attempts += 1
            else:
                # Revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                swap_attempts += 1
        else:
            # No candidate found, revert
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
            swap_attempts += 1

    # Final cleanup
    try_remove_redundant()

    return [list(idx_to_word(c)) for c in code_list]
