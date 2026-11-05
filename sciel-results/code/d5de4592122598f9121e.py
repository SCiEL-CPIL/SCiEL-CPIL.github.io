import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Fallback for general cases
    if q != 3 or n != 8 or radius != 1:
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
    start_time = time.time()
    deadline = start_time + 115  # Safe margin

    # Precompute neighbors
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    neighbors_np = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(8):
            orig_val = w[i]
            for val in range(3):
                if val != orig_val:
                    nw = w[:]
                    nw[i] = val
                    new_idx = 0
                    for v in nw:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors_np[idx] = np.array(nbrs, dtype=np.int32)

    # Greedy Construction (vectorized)
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_list = []
    code_set = set()
    
    # Build adjacency matrix (sparse) for fast gain computation
    # Each codeword covers 17 positions (itself + 16 neighbors)
    # Use a flat list of (row, col) pairs for the adjacency
    adj_rows = []
    adj_cols = []
    for i in range(total_words):
        for nb in neighbors_np[i]:
            adj_rows.append(i)
            adj_cols.append(nb)
    adj_rows = np.array(adj_rows, dtype=np.int32)
    adj_cols = np.array(adj_cols, dtype=np.int32)
    
    while np.any(uncovered_mask) and time.time() < deadline - 60:
        # Vectorized gain: for each candidate i, count uncovered neighbors
        # gains[i] = sum(uncovered_mask[neighbors_np[i]])
        # Using the adjacency: for each edge (i, nb), if uncovered[nb], add 1 to gains[i]
        # This is equivalent to: gains = sum over edges where col is uncovered
        mask_col = uncovered_mask[adj_cols]
        gains = np.bincount(adj_rows[mask_col], minlength=total_words)
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        uncovered_mask[neighbors_np[best_idx]] = False

    if np.any(uncovered_mask):
        holes = np.where(uncovered_mask)[0]
        for h in holes:
            if h not in code_set:
                code_set.add(h)
                code_list.append(h)
                uncovered_mask[neighbors_np[h]] = False

    def local_search(code_list, code_set, cover_count, deadline):
        # Remove redundant
        def is_redundant(c):
            return np.min(cover_count[neighbors_np[c]]) >= 2

        def remove_redundant():
            changed = False
            for _ in range(10):
                progress = False
                for i in range(len(code_list) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        return changed
                    c = code_list[i]
                    if is_redundant(c):
                        code_list.pop(i)
                        code_set.discard(c)
                        cover_count[neighbors_np[c]] -= 1
                        changed = True
                        progress = True
                if not progress:
                    break
            return changed

        remove_redundant()

        while time.time() < deadline - 5:
            if remove_redundant():
                continue

            # Try 2-out, 1-in (size reduction)
            improved = False
            # Sample pairs to check
            if len(code_list) > 2:
                # Random sampling of pairs
                for _ in range(100): # Try 100 random pairs
                    if time.time() > deadline - 5:
                        break
                    i1, i2 = random.sample(range(len(code_list)), 2)
                    c1 = code_list[i1]
                    c2 = code_list[i2]
                    
                    # Temporarily remove
                    cover_count[neighbors_np[c1]] -= 1
                    cover_count[neighbors_np[c2]] -= 1
                    
                    holes_mask = cover_count == 0
                    holes = np.where(holes_mask)[0]
                    
                    if len(holes) <= 1:
                        # Find replacement
                        cand_pool = set()
                        for h in holes:
                            for nb in neighbors_np[h]:
                                if nb not in code_set:
                                    cand_pool.add(nb)
                        
                        if len(holes) == 0:
                            # Just keep them removed
                            code_list.remove(c1)
                            code_list.remove(c2)
                            code_set.discard(c1)
                            code_set.discard(c2)
                            improved = True
                            break
                        elif len(holes) == 1:
                            h = holes[0]
                            best_cand = -1
                            best_score = -1
                            for cand in cand_pool:
                                if h in neighbors_np[cand]:
                                    score = np.sum(cover_count[neighbors_np[cand]] >= 1)
                                    if score > best_score:
                                        best_score = score
                                        best_cand = cand
                            if best_cand != -1:
                                # Remove c1, c2, add best_cand
                                code_list.remove(c1)
                                code_list.remove(c2)
                                code_set.discard(c1)
                                code_set.discard(c2)
                                code_list.append(best_cand)
                                code_set.add(best_cand)
                                cover_count[neighbors_np[best_cand]] += 1
                                improved = True
                                break
                    # Revert
                    cover_count[neighbors_np[c1]] += 1
                    cover_count[neighbors_np[c2]] += 1

            if improved:
                continue

            # Try 1-out, 1-in (structure change)
            if len(code_list) > 1:
                c = random.choice(code_list)
                cover_count[neighbors_np[c]] -= 1
                holes_mask = cover_count == 0
                holes = np.where(holes_mask)[0]
                
                if len(holes) <= 1:
                    cand_pool = set()
                    for h in holes:
                        for nb in neighbors_np[h]:
                            if nb not in code_set:
                                cand_pool.add(nb)
                    
                    if len(holes) == 0:
                        # Redundant, should have been removed
                        code_list.remove(c)
                        code_set.discard(c)
                    elif len(holes) == 1:
                        h = holes[0]
                        best_cand = -1
                        best_score = -1
                        for cand in cand_pool:
                            if h in neighbors_np[cand]:
                                score = np.sum(cover_count[neighbors_np[cand]] >= 1)
                                if score > best_score:
                                    best_score = score
                                    best_cand = cand
                        if best_cand != -1:
                            code_list.remove(c)
                            code_set.discard(c)
                            code_list.append(best_cand)
                            code_set.add(best_cand)
                            cover_count[neighbors_np[best_cand]] += 1
                # Revert if not improved (size didn't change, but structure did)
                # Actually, we always revert if we didn't break out with an improvement?
                # No, 1-swap is for escaping local minima. We accept it if it doesn't increase size.
                # But we need to be careful not to loop infinitely.
                # Let's just revert if we didn't find a valid replacement.
                if len(holes) > 1 or (len(holes) == 1 and best_cand == -1):
                    cover_count[neighbors_np[c]] += 1
                # If we found a replacement, we kept it.

        return code_list, code_set, cover_count

    # Initial Local Search
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_np[c]] += 1
        
    best_code_list, best_code_set, best_cover_count = local_search(code_list, code_set, cover_count, deadline)
    best_size = len(best_code_list)

    # Iterated Local Search
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1
        # Perturb best code
        perturbed_list = list(best_code_list)
        perturbed_set = set(perturbed_list)
        perturbed_cover = np.zeros(total_words, dtype=np.int32)
        for c in perturbed_list:
            perturbed_cover[neighbors_np[c]] += 1
        
        # Identify redundant codewords (those whose removal doesn't create holes)
        redundant = []
        for i, c in enumerate(perturbed_list):
            if np.min(perturbed_cover[neighbors_np[c]]) >= 2:
                redundant.append(i)
        
        # Perturbation strategy: prefer removing redundant codewords
        num_swaps = max(1, int(0.03 * len(perturbed_list)))
        if redundant:
            # Remove some redundant codewords and replace with random ones
            num_from_redundant = min(num_swaps, len(redundant))
            for _ in range(num_from_redundant):
                idx_to_remove = random.choice(redundant)
                redundant.remove(idx_to_remove)
                c_to_remove = perturbed_list[idx_to_remove]
                perturbed_list.pop(idx_to_remove)
                perturbed_set.discard(c_to_remove)
                perturbed_cover[neighbors_np[c_to_remove]] -= 1
                
                # Replace with a random word not in set
                # Use rejection sampling to avoid building full candidate list
                for _attempt in range(100):
                    c_to_add = random.randint(0, total_words - 1)
                    if c_to_add not in perturbed_set:
                        perturbed_list.append(c_to_add)
                        perturbed_set.add(c_to_add)
                        perturbed_cover[neighbors_np[c_to_add]] += 1
                        break
        
        # Fill holes in perturbed code before local search
        uncovered_mask = perturbed_cover == 0
        if np.any(uncovered_mask):
            holes = np.where(uncovered_mask)[0]
            for h in holes:
                # Find a neighbor not in set
                found = False
                for nb in neighbors_np[h]:
                    if nb not in perturbed_set:
                        perturbed_list.append(int(nb))
                        perturbed_set.add(int(nb))
                        perturbed_cover[neighbors_np[nb]] += 1
                        found = True
                        break
                if not found:
                    if h not in perturbed_set:
                        perturbed_list.append(int(h))
                        perturbed_set.add(int(h))
                        perturbed_cover[neighbors_np[h]] += 1

        new_list, new_set, new_cover = local_search(perturbed_list, perturbed_set, perturbed_cover, deadline)
        
        if len(new_list) < best_size:
            best_code_list = new_list
            best_code_set = new_set
            best_cover_count = new_cover
            best_size = len(new_list)

    # Final Validity Check
    final_cover = np.zeros(total_words, dtype=bool)
    for c in best_code_list:
        final_cover[neighbors_np[c]] = True
        
    if not np.all(final_cover):
        uncovered_idx = np.where(~final_cover)[0]
        for idx in uncovered_idx:
             if idx not in best_code_set:
                 best_code_list.append(idx)
                 best_code_set.add(idx)

    # Final polish: try exhaustive 2-out/1-in moves with time budget
    if time.time() < deadline - 3:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code_list:
            cover_count[neighbors_np[c]] += 1
        
        # First pass: remove all redundancies
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for i in range(len(best_code_list) - 1, -1, -1):
                c = best_code_list[i]
                if np.min(cover_count[neighbors_np[c]]) >= 2:
                    best_code_list.pop(i)
                    best_code_set.discard(c)
                    cover_count[neighbors_np[c]] -= 1
                    changed = True
        
        # Try 2-out/1-in moves
        if len(best_code_list) > 2 and time.time() < deadline - 3:
            n_codes = len(best_code_list)
            # Try pairs systematically but with time limit
            tried = 0
            max_tries = 50000
            for i in range(n_codes):
                if time.time() > deadline - 3:
                    break
                if tried >= max_tries:
                    break
                c1 = best_code_list[i]
                for j in range(i + 1, n_codes):
                    if time.time() > deadline - 3:
                        break
                    if tried >= max_tries:
                        break
                    tried += 1
                    c2 = best_code_list[j]
                    
                    cover_count[neighbors_np[c1]] -= 1
                    cover_count[neighbors_np[c2]] -= 1
                    
                    holes = np.where(cover_count == 0)[0]
                    
                    if len(holes) == 0:
                        # Both redundant, remove both
                        best_code_list.pop(j)
                        best_code_list.pop(i)
                        best_code_set.discard(c1)
                        best_code_set.discard(c2)
                        # Rebuild cover_count
                        cover_count = np.zeros(total_words, dtype=np.int32)
                        for c in best_code_list:
                            cover_count[neighbors_np[c]] += 1
                        break
                    elif len(holes) == 1:
                        h = int(holes[0])
                        # Find best replacement
                        best_cand = -1
                        best_score = -1
                        for nb in neighbors_np[h]:
                            if nb not in best_code_set:
                                score = int(np.sum(cover_count[neighbors_np[nb]] >= 1))
                                if score > best_score:
                                    best_score = score
                                    best_cand = int(nb)
                        if best_cand != -1:
                            best_code_list.pop(j)
                            best_code_list.pop(i)
                            best_code_set.discard(c1)
                            best_code_set.discard(c2)
                            best_code_list.append(best_cand)
                            best_code_set.add(best_cand)
                            cover_count = np.zeros(total_words, dtype=np.int32)
                            for c in best_code_list:
                                cover_count[neighbors_np[c]] += 1
                            break
                    else:
                        cover_count[neighbors_np[c1]] += 1
                        cover_count[neighbors_np[c2]] += 1
                        continue
                    
                    # Revert if we didn't break
                    cover_count[neighbors_np[c1]] += 1
                    cover_count[neighbors_np[c2]] += 1
                else:
                    # Inner loop broke, check if we improved
                    if len(best_code_list) < n_codes - 1:
                        break

    return [idx_to_word(c) for c in best_code_list]
