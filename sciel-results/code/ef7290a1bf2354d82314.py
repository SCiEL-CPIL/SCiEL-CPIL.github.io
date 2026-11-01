import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17
    
    # Precompute neighbors
    neighbors = np.zeros((total, num_neighbors), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1

    best_size = total
    best_set = None
    
    # We will run multiple iterations of: Greedy -> Local Search
    # The key improvement from inspiration is a more efficient swap/local search
    # using candidate intersection instead of full vectorized checks over all 6561 words.
    
    while time.time() < deadline - 3.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                break
            
            counts = uncovered_mask[neighbors].sum(axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
        
        if uncovered_count > 0:
            continue
        
        # 2. Local Search (Removal + Swaps)
        changed = True
        local_iter = 0
        while changed and time.time() < deadline - 3.0:
            changed = False
            local_iter += 1
            
            # Pass 1: Removal
            # Check if any codeword is redundant (all its neighbors have count > 1)
            # We iterate in random order to break symmetry
            indices_to_check = list(range(len(selected)))
            np.random.shuffle(indices_to_check)
            
            removed_any = False
            for i in indices_to_check:
                if time.time() > deadline - 5.0:
                    break
                cw = selected[i]
                # If min count in its ball is > 1, it's redundant
                if np.min(cover_count[neighbors[cw]]) > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    removed_any = True
                    break # Restart removal pass
            
            if not changed and len(selected) > 486:
                # Pass 2: Swaps (Perturbation)
                # Try to replace a codeword to potentially allow further removals
                # or to find a better local minimum.
                # Strategy: Pick a random codeword, remove it, find a replacement that covers
                # all the "critical" words (words that would become uncovered).
                
                if time.time() < deadline - 10.0:
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    idx_to_remove = np.random.randint(len(selected))
                    cw_removed = selected.pop(idx_to_remove)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) > 0:
                        # Find replacement using intersection of neighbor sets
                        # A valid replacement must cover all words in U.
                        # The set of codewords covering a word v is neighbors[v] (by symmetry).
                        # So we need the intersection of neighbors[v] for all v in U.
                        
                        # Start with the set of neighbors of the first uncovered word
                        # Convert to a set for fast intersection
                        # neighbors[U[0]] is an array of ints
                        candidate_set = set(neighbors[U[0]].tolist())
                        
                        # Intersect with neighbors of other uncovered words
                        for u in U[1:]:
                            if not candidate_set:
                                break
                            candidate_set &= set(neighbors[u].tolist())
                        
                        # Exclude currently selected codewords (and the one we removed? 
                        # Actually, we can re-add the removed one, but that's a no-op swap.
                        # Usually we want a NEW codeword to potentially change the structure.
                        # But allowing the same one back is safe if it's the only option.
                        # However, if we just put it back, we haven't changed the set.
                        # Let's exclude selected ones to force a change, unless no other option.
                        
                        selected_mask = np.zeros(total, dtype=bool)
                        selected_mask[selected] = True
                        
                        valid_candidates = []
                        for c in candidate_set:
                            if not selected_mask[c]:
                                valid_candidates.append(c)
                        
                        if len(valid_candidates) > 0:
                            # Pick one randomly
                            cw2 = valid_candidates[np.random.randint(len(valid_candidates))]
                            selected.append(cw2)
                            cover_count[neighbors[cw2]] += 1
                            changed = True # Allow further removals in next iteration
                        else:
                            # No replacement found, revert
                            selected = orig_selected
                            cover_count = orig_cover_count
                    else:
                        # U is empty, removal was valid (should have been caught in removal pass)
                        # But if it happens, we improved size.
                        changed = True

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        if best_size <= 486:
            break

    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
