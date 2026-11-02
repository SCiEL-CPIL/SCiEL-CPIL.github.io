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
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
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
    
    # Optimization: Precompute neighbor indices for fast access
    # We will use cover_count array of size 'total'
    
    while time.time() < deadline - 5.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # Precompute counts of uncovered neighbors for each word? 
        # No, we update dynamically.
        
        while uncovered_count > 0:
            # Vectorized count of uncovered words in each ball
            # neighbors is (total, 17), uncovered_mask is (total,)
            # We want sum of uncovered_mask[neighbors[i]] for each i
            counts = np.zeros(total, dtype=np.int32)
            # This loop is the bottleneck of greedy. 
            # Alternative: maintain counts incrementally? Hard with random selection.
            # Let's try to speed up the counting.
            # counts[i] = sum(uncovered_mask[neighbors[i]])
            # This is essentially a sparse matrix-vector product if neighbors were sparse matrix.
            # Since neighbors is dense (17 cols), we can do:
            # counts = neighbors @ uncovered_mask ? No, neighbors contains indices.
            # We need: for each i, sum(uncovered_mask[neighbors[i, :]])
            # This is hard to vectorize directly without advanced indexing which is slow for large N.
            # However, N=6561 is small. 
            # Let's use the fact that we only need the MAX.
            
            # Fast path: 
            # If we just pick a random word, it's slow.
            # Let's stick to the max strategy but optimize the counting.
            # We can compute counts by iterating over uncovered words and incrementing their neighbors?
            # That's O(U * 17). If U is large, it's slow.
            # Current method is O(T * 17).
            
            # Let's try a different greedy strategy: 
            # Start with all words, remove those with min count? No, that's local search.
            
            # Let's stick to the current greedy but ensure it's as fast as possible.
            # Using `uncovered_mask[neighbors].sum(axis=1)` is actually quite fast in NumPy for 6561x17.
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
            
            if time.time() > deadline - 10.0:
                break
        
        if uncovered_count > 0:
            continue
        
        # 2. Optimized Local Search
        # We maintain 'selected' as a list, but we also keep a 'in_selected' boolean mask for O(1) check
        in_selected = np.zeros(total, dtype=bool)
        in_selected[selected] = True
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            
            # Pass 1: Removal
            # A word i is removable if min(cover_count[neighbors[i]]) > 1
            # We can compute this for all selected words at once.
            # min_counts[i] = min(cover_count[neighbors[i]])
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find removable words among selected
            # removable_mask[i] = in_selected[i] and min_counts[i] > 1
            removable_mask = in_selected & (min_counts > 1)
            
            if np.any(removable_mask):
                # Prioritize removing words with highest min_count (most redundant)
                # We want to pick one or more. Let's pick the one with highest min_count to maximize potential gain?
                # Or just pick one randomly from removable to allow exploration?
                # Greedy removal: pick the one that frees the most "slots"?
                # Actually, removing any valid word reduces size by 1.
                # To help future swaps, maybe remove one that is "least useful" for covering unique words?
                # min_counts > 1 means no word is uniquely covered by it.
                # Let's just pick one randomly from the removable set to escape local minima.
                removable_indices = np.where(removable_mask)[0]
                idx_to_remove = removable_indices[np.random.randint(len(removable_indices))]
                
                # Remove it
                selected.remove(idx_to_remove)
                in_selected[idx_to_remove] = False
                cover_count[neighbors[idx_to_remove]] -= 1
                changed = True
                continue # Restart local search loop to re-evaluate
            
            # Pass 2: Swaps (Perturbation)
            # If no removals, try to swap.
            # Pick a random selected word, remove it, see if we can replace it with a non-selected word.
            if len(selected) > 486: # Only bother if we are not at theoretical best
                idx_to_remove = np.random.randint(len(selected))
                cw_removed = selected[idx_to_remove]
                
                # Temporarily remove
                selected.pop(idx_to_remove)
                in_selected[cw_removed] = False
                cover_count[neighbors[cw_removed]] -= 1
                
                # Find uncovered words
                U = np.where(cover_count == 0)[0]
                
                if len(U) > 0:
                    # We need a word c not in selected (and c != cw_removed) such that U is subset of neighbors[c]
                    # Condition: for all u in U, u is in neighbors[c]
                    # Equivalent: sum( (neighbors[c] == u) for all u in U ) >= 1 for each u? No.
                    # Equivalent: For each c, count how many words in U are covered by c.
                    # If count == len(U), then c covers all U.
                    
                    # Vectorized check:
                    # We want to compute, for all c in range(total), the number of u in U such that u is in neighbors[c].
                    # This is the same as: for each u in U, mark neighbors of u? No.
                    # neighbors[c] is the ball of c.
                    # We want to know if U is a subset of Ball(c).
                    
                    # Let's compute `covers_U` array of size total.
                    # covers_U[c] = 0
                    # for u in U:
                    #    covers_U[neighbors_of_u] += 1  <-- Wait, neighbors is defined by center.
                    # neighbors[c] contains words at dist 0 or 1 from c.
                    # So u is in Ball(c) iff c is in Ball(u).
                    # Yes! Symmetry of Hamming distance.
                    
                    # So:
                    # 1. Initialize covers_U = zeros(total)
                    # 2. For each u in U, increment covers_U[neighbors[u]]
                    # 3. Valid candidates are c where covers_U[c] == len(U) and not in_selected[c]
                    
                    covers_U = np.zeros(total, dtype=np.int32)
                    for u in U:
                        covers_U[neighbors[u]] += 1
                    
                    # Find candidates
                    # Condition: covers_U[c] == len(U) and not in_selected[c]
                    # Note: in_selected is False for cw_removed, so it might be a candidate if it covers U?
                    # But we just removed it, so it didn't cover U (that's why U is uncovered).
                    # So cw_removed will have covers_U[cw_removed] < len(U).
                    
                    valid_mask = (covers_U == len(U)) & (~in_selected)
                    valid_candidates = np.where(valid_mask)[0]
                    
                    if len(valid_candidates) > 0:
                        # Pick one randomly
                        cw2 = valid_candidates[np.random.randint(len(valid_candidates))]
                        selected.append(cw2)
                        in_selected[cw2] = True
                        cover_count[neighbors[cw2]] += 1
                        changed = True # Allow further removals
                    else:
                        # Revert
                        selected.insert(idx_to_remove, cw_removed)
                        in_selected[cw_removed] = True
                        cover_count[neighbors[cw_removed]] += 1
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
