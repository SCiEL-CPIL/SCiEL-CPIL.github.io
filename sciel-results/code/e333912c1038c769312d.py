import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors efficiently using vectorization
    # neighbors[w] contains w and all words at distance 1
    # There are 1 + 2*8 = 17 neighbors per word
    
    # Build neighbors table: (total, 17) int32 array
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    # For each word, compute its 16 neighbors (distance 1) + itself
    # Use vectorized approach: for each position and each digit change
    positions = np.arange(n)
    weights = 3 ** (n - 1 - positions)  # [1, 3, 9, 27, 81, 243, 729, 2187]
    weights = weights.astype(np.int32)
    
    # For word w, digit at position p is (w // weights[p]) % 3
    # Changing to digit d gives w + (d - digit) * weights[p]
    
    # Build all neighbors in a vectorized way
    # For each of the 16 (position, new_digit) combinations where new_digit != current digit
    # Actually, for each position p and each of 2 other digits, we get a neighbor
    
    # Let's build it: for position p, the 2 neighbors are w + (d1 - digit)*w_p and w + (d2 - digit)*w_p
    # where d1, d2 are the two digits different from the current digit
    
    # Simpler: just loop over positions and compute
    for p in range(n):
        w_p = weights[p]
        # For each word, get its digit at position p
        digits_p = (np.arange(total) // w_p) % 3
        # The two other digits
        for d in range(3):
            mask = digits_p != d
            # For words where digit != d, adding (d - digit)*w_p gives neighbor with digit d
            neighbors[mask, 1 + 2*p] = (np.arange(total)[mask] + (d - digits_p[mask]) * w_p).astype(np.int32)
    
    # Wait, I need to fill all 16 neighbor slots properly
    # Let me redo this more carefully
    
    neighbors = np.zeros((total, 17), dtype=np.int32)
    neighbors[:, 0] = np.arange(total)  # self
    
    neighbor_idx = 1
    for p in range(n):
        w_p = weights[p]
        digits_p = (np.arange(total) // w_p) % 3
        for d in range(3):
            mask = digits_p != d
            if mask.any():
                new_vals = (np.arange(total)[mask] + (d - digits_p[mask]) * w_p).astype(np.int32)
                neighbors[mask, neighbor_idx] = new_vals
            neighbor_idx += 1
    
    # Verify: should be exactly 17 columns filled
    assert neighbor_idx == 17, f"neighbor_idx = {neighbor_idx}"
    
    # Now run multiple greedy+local search iterations
    best_size = total
    best_set = None
    best_cover_count = None
    
    rng = np.random.RandomState(42)
    
    max_iterations = 50
    iteration = 0
    
    while time.time() < deadline - 3.0 and iteration < max_iterations:
        iteration += 1
        
        # Greedy set cover
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            counts = uncovered_mask[neighbors].sum(axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            # Tie-break: prefer words with lower index for determinism, or random
            if len(candidates) > 1:
                best_idx = candidates[rng.randint(len(candidates))]
            else:
                best_idx = candidates[0]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            if time.time() > deadline - 5.0:
                break
        
        if uncovered_count > 0:
            # Incomplete cover, try to finish it quickly
            for w in range(total):
                if uncovered_mask[w]:
                    selected.append(w)
                    cover_count[neighbors[w]] += 1
                    uncovered_mask[neighbors[w]] = False
                    uncovered_count -= np.sum(uncovered_mask[neighbors[w]] & (cover_count[neighbors[w]] == 0))
                    if time.time() > deadline - 5.0:
                        break
        
        if uncovered_count > 0:
            continue
        
        # Local search: iterative improvement
        changed = True
        local_iter = 0
        while changed and time.time() < deadline - 2.0:
            changed = False
            local_iter += 1
            
            # Pass 1: Try to remove redundant codewords
            # A codeword is removable if every word in its ball is covered by at least one other codeword
            # i.e., min cover_count over its ball > 1
            
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Try removals in random order
            indices_to_check = list(range(len(selected)))
            rng.shuffle(indices_to_check)
            
            for i in indices_to_check:
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    break
            
            # Pass 2: Try swaps - remove one, add another that covers newly uncovered
            if not changed and len(selected) > 486:
                # Try a few random swaps
                for _ in range(3):
                    if time.time() > deadline - 3.0:
                        break
                    
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    idx_to_remove = rng.randint(len(selected))
                    cw_removed = selected.pop(idx_to_remove)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    # Find uncovered words
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) == 0:
                        # Removal was valid, we improved
                        changed = True
                        break
                    elif len(U) <= 17:
                        # Find a replacement that covers all of U
                        # U is subset of neighbors[cw2] for some cw2
                        is_valid = np.ones(total, dtype=bool)
                        for u in U:
                            mask_u = (neighbors == u).any(axis=1)
                            is_valid &= mask_u
                        
                        # Exclude already selected
                        selected_mask = np.zeros(total, dtype=bool)
                        selected_mask[selected] = True
                        is_valid[selected_mask] = False
                        
                        valid_candidates = np.where(is_valid)[0]
                        if len(valid_candidates) > 0:
                            cw2 = valid_candidates[rng.randint(len(valid_candidates))]
                            selected.append(cw2)
                            cover_count[neighbors[cw2]] += 1
                            changed = True
                            break
                        else:
                            # Revert
                            selected = orig_selected
                            cover_count = orig_cover_count
                    else:
                        # Too many uncovered, revert
                        selected = orig_selected
                        cover_count = orig_cover_count
            
            # Pass 3: Perturbation - remove a random codeword and re-add greedily
            if not changed and len(selected) > 486 and time.time() < deadline - 5.0:
                orig_selected = selected.copy()
                orig_cover_count = cover_count.copy()
                
                # Remove 2-3 random codewords
                num_remove = min(3, len(selected))
                to_remove = rng.choice(len(selected), size=num_remove, replace=False)
                for idx in sorted(to_remove, reverse=True):
                    cw = selected.pop(idx)
                    cover_count[neighbors[cw]] -= 1
                
                # Re-add greedily until covered
                uncovered_mask = (cover_count == 0)
                uncovered_count = np.sum(uncovered_mask)
                
                while uncovered_count > 0:
                    counts = uncovered_mask[neighbors].sum(axis=1)
                    max_count = np.max(counts)
                    if max_count == 0:
                        break
                    candidates = np.where(counts == max_count)[0]
                    best_idx = candidates[rng.randint(len(candidates))]
                    selected.append(best_idx)
                    ball = neighbors[best_idx]
                    was_uncovered = uncovered_mask[ball]
                    num_newly_covered = np.sum(was_uncovered)
                    uncovered_count -= num_newly_covered
                    cover_count[ball] += 1
                    uncovered_mask[ball] = False
                    
                    if time.time() > deadline - 4.0:
                        break
                
                if uncovered_count == 0 and len(selected) < len(orig_selected):
                    changed = True
                else:
                    # Revert
                    selected = orig_selected
                    cover_count = orig_cover_count
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            best_cover_count = cover_count.copy()
            
            if best_size <= 486:
                break
    
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
    
    # Final polish: try more aggressive removals
    if time.time() < deadline - 2.0 and best_size > 486:
        cover_count = best_cover_count.copy()
        selected = best_set.copy()
        
        # Multiple passes of removal
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            min_counts = np.min(cover_count[neighbors], axis=1)
            indices_to_check = list(range(len(selected)))
            rng.shuffle(indices_to_check)
            improved = False
            for i in indices_to_check:
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    improved = True
            if not improved:
                break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected
    
    # Convert to list of digit lists
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
