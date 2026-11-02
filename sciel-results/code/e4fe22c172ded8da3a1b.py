import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
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
    
    # Precompute neighbors
    # We use a flat array representation for speed
    # neighbors[cw] = list of 17 indices (cw itself + 16 neighbors)
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    # Vectorized precomputation of neighbors
    # This is a one-time cost, so a loop is fine and likely faster than complex vectorization for 6561 items
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
                
    # Precompute in_ball matrix for fast subset checks
    # in_ball[w, cw] is True if w is in ball(cw)
    in_ball = np.zeros((total, total), dtype=bool)
    for cw in range(total):
        in_ball[neighbors[cw], cw] = True

    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search (Removal + Swap)
    # The local search with swaps is key to escaping local optima and reaching 486.
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        # To get diversity, we can randomize the tie-breaking in greedy.
        # Or simply run greedy with a random starting point? No, greedy starts with empty set.
        # We can randomize the order of candidates if counts are equal.
        
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        selected_set = set()
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Optimization: Instead of checking np.any(uncovered_mask) which is O(N),
        # we can maintain a count of uncovered words.
        uncovered_count = total
        
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            # uncovered_mask[neighbors] shape (total, 17)
            # sum along axis 1
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            # Find max count
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            # Find all candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Pick one randomly to add diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            # Update cover counts and uncovered mask
            # We need to update uncovered_count carefully
            ball = neighbors[best_idx]
            # Words in ball that were uncovered
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        if uncovered_count > 0:
            # If we broke early due to time, this set might be invalid.
            # We should only consider it if it's valid.
            # Since we broke, it's likely invalid. Skip this iteration.
            continue
        
        # 2. Local Search: Removal and Swap
        # We maintain cover_count and the list of selected codewords.
        # Removal: If min(cover_count[neighbors[cw]]) > 1, remove cw.
        # Swap: If removing cw makes some words uncovered, try to add another cw2 that covers them.
        # A simpler swap: 
        #   1. Identify "critical" words (cover_count == 1).
        #   2. A codeword cw is "removable" if it doesn't cover any critical word.
        #   3. If no codeword is removable, try a swap:
        #      Pick a codeword cw that covers a critical word.
        #      Remove it (temporarily).
        #      Find the set of words that become uncovered (they were covered only by cw).
        #      Find a codeword cw2 (not in set) that covers all these words.
        #      If found, add cw2.
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal
            # We can vectorize the check for removability somewhat, but let's stick to loop for clarity and low constant overhead.
            # Actually, checking min(cover_count[neighbors[cw]]) for all cw is O(N * 17).
            # min_counts = np.min(cover_count[neighbors], axis=1)
            # removable_mask = min_counts > 1
            # This is very fast with NumPy.
            
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # We only care about indices in 'selected'
            # Since selected is a list, we can just iterate.
            # But if we remove many, we should update selected.
            
            removed_any = False
            # Iterate through selected in random order to break symmetry
            indices_to_check = list(range(len(selected)))
            np.random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                cw = selected[i]
                if min_counts[cw] > 1:
                    # Remove it
                    selected.pop(i)
                    selected_set.remove(cw)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    removed_any = True
                    break # Restart loop to recompute min_counts
            
            if not removed_any:
                # No removals possible. Try Swaps.
                if time.time() < deadline - 10.0 and len(selected) > 486:
                    # Perturbation: Remove a random codeword, then repair.
                    orig_selected = selected.copy()
                    orig_selected_set = selected_set.copy()
                    orig_cover_count = cover_count.copy()
                    
                    # Pick a random codeword to remove
                    idx_to_remove = np.random.randint(len(selected))
                    cw_removed = selected[idx_to_remove]
                    selected.pop(idx_to_remove)
                    selected_set.remove(cw_removed)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    # Now we have uncovered words (cover_count == 0)
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) == 0:
                        # Removal was valid, we improved!
                        changed = True
                        # Don't break, try to remove more
                        continue 
                    else:
                        # Try to find a replacement
                        # Optimization: Only check candidates that cover U[0]
                        # Candidates are in neighbors[U[0]]
                        replacement_found = False
                        for cw2 in neighbors[U[0]]:
                            if cw2 in selected_set:
                                continue
                            
                            # Check if cw2 covers all of U using in_ball matrix
                            if in_ball[U, cw2].all():
                                selected.append(cw2)
                                selected_set.add(cw2)
                                cover_count[neighbors[cw2]] += 1
                                replacement_found = True
                                changed = True
                                break
                        
                        if not replacement_found:
                            # Revert
                            selected = orig_selected
                            selected_set = orig_selected_set
                            cover_count = orig_cover_count
                            # Break local search to try new greedy
                            break
        # End Local Search

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        if best_size <= 486:
            # We found a very good solution.
            # We can stop early.
            break

    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback
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
