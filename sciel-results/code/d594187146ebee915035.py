import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def is_valid_code(code_indices):
        if not code_indices:
            return total_points == 0
        is_covered = np.zeros(total_points, dtype=bool)
        for idx in code_indices:
            is_covered[neighbors[idx]] = True
        return np.all(is_covered)

    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            uncovered = ~is_covered
            counts = np.sum(uncovered[neighbors], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break

            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]

            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True

        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []

        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)

        for idx in selected_set:
            coverage_count[neighbors[idx]] += 1

        changed = True
        while changed:
            changed = False
            sel_list = list(selected_set)
            to_remove = []
            for idx in sel_list:
                if np.all(coverage_count[neighbors[idx]] > 1):
                    to_remove.append(idx)
            if to_remove:
                changed = True
                for rem_idx in to_remove:
                    coverage_count[neighbors[rem_idx]] -= 1
                    selected_set.discard(rem_idx)

        return list(selected_set)

    def local_search_v2(selected_indices, deadline):
        """
        Improved local search with bit-parallel coverage deficit representation.
        Uses Python integers as bitsets for 6561 points.
        """
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)

        # Bit-parallel representation
        # uncovered_bits: bit i is 1 if point i is uncovered
        # For each codeword c, neighbor_mask[c] is a big int with bits set for all points in ball(c)
        
        # Pre-compute neighbor masks as Python integers (big integers)
        # This is the key speedup: bitwise operations on 6561 bits are very fast in Python
        
        neighbor_masks = []
        for idx in range(total_points):
            mask = 0
            for p in neighbors[idx]:
                mask |= (1 << p)
            neighbor_masks.append(mask)
        
        # Convert to list for faster access
        # Actually, let's use a numpy array of Python ints? No, just a list.
        
        # Current uncovered set as a big integer
        uncovered_bits = (1 << total_points) - 1  # All points initially uncovered
        
        # But wait, we start with a code that covers everything.
        # Let's recompute uncovered from the current code.
        # Actually, prune ensures the code is valid, so uncovered should be 0.
        # Let's verify.
        covered_mask = 0
        for idx in selected_list:
            covered_mask |= neighbor_masks[idx]
        uncovered_bits = ((1 << total_points) - 1) & ~covered_mask
        
        # If uncovered_bits != 0, the code is invalid. But prune should ensure validity.
        # For safety, if invalid, we fall back to greedy.
        if uncovered_bits != 0:
            # This shouldn't happen if prune is correct, but let's handle it
            # Re-run greedy
            selected_list = greedy_cover(0)
            selected_list = prune(selected_list)
            selected_set = set(selected_list)
            covered_mask = 0
            for idx in selected_list:
                covered_mask |= neighbor_masks[idx]
            uncovered_bits = ((1 << total_points) - 1) & ~covered_mask
        
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        def reduce_code_bitwise(code, uncovered):
            """Remove redundant codewords using bit operations."""
            if not code:
                return code, uncovered
            current_code = list(code)
            current_uncovered = uncovered
            
            while True:
                removed_any = False
                next_code = []
                for c in current_code:
                    # Check if removing c leaves any point uncovered
                    # If (current_uncovered | ~neighbor_masks[c]) == current_uncovered, 
                    # then c is redundant (all its points are covered by others)
                    # Wait, that's not right.
                    # A point p is uniquely covered by c if p is in ball(c) and p is not in ball(c') for any other c'.
                    # In bit terms: unique_points = neighbor_masks[c] & ~ (OR of all other masks)
                    # This is expensive to compute naively.
                    
                    # Alternative: 
                    # Let's compute the mask of points covered by current_code \ {c}
                    # This is expensive if done from scratch.
                    
                    # Let's use a different approach for reduction:
                    # For each c, check if neighbor_masks[c] is a subset of (covered_mask ^ neighbor_masks[c])
                    # i.e., if all points in c's ball are covered by other codewords.
                    
                    # Let's maintain a coverage_count array instead for reduction, it's simpler.
                    pass
                
                # Let's stick with the coverage_count approach for reduction, it's more straightforward.
                # The bit-parallel approach is mainly for the swap feasibility check.
                break
                
            return current_code, current_uncovered
        
        # Let's use a hybrid approach:
        # Use coverage_count for reduction (it's fast enough with numpy)
        # Use bit-parallel for swap feasibility checks
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        def reduce_code(code, counts):
            """Remove redundant codewords iteratively."""
            if not code:
                return code, counts
            current_code = list(code)
            current_counts = counts.copy()
            
            while True:
                removed_any = False
                next_code = []
                for c in current_code:
                    if np.min(current_counts[neighbors[c]]) >= 2:
                        current_counts[neighbors[c]] -= 1
                        removed_any = True
                    else:
                        next_code.append(c)
                current_code = next_code
                if not removed_any:
                    break
            return current_code, current_counts
        
        # Pre-compute complement masks for each codeword
        # complement_masks[c] = ~neighbor_masks[c] & ((1 << total_points) - 1)
        # This represents points NOT covered by codeword c
        all_mask = (1 << total_points) - 1
        complement_masks = []
        for idx in range(total_points):
            complement_masks.append(all_mask & ~neighbor_masks[idx])
        
        while time.time() < deadline:
            # 1. Reduction Pass
            old_size = len(selected_list)
            selected_list, coverage_count = reduce_code(selected_list, coverage_count)
            
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
            
            if time.time() > deadline:
                break
            
            if len(selected_list) == 0:
                break
            
            # 2. Swap Pass (1-for-1) using bit-parallel feasibility check
            improved = False
            indices = list(range(len(selected_list)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                c = selected_list[idx]
                u_indices = neighbors[c]
                unique_mask = coverage_count[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    pos = selected_list.index(c)
                    selected_list.pop(pos)
                    selected_set.discard(c)
                    coverage_count[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 4:
                    continue
                
                # Find candidates that cover all unique points
                # Using bit-parallel: 
                # A candidate c' is feasible if it covers all unique_points
                # i.e., (neighbor_masks[c'] & unique_points_mask) == unique_points_mask
                # where unique_points_mask is a big int with bits set for unique_points
                
                unique_points_mask = 0
                for p in unique_points:
                    unique_points_mask |= (1 << p)
                
                # Find candidates by intersecting neighbor lists of unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                candidates_set.difference_update(selected_set)
                if not candidates_set:
                    continue
                
                # Score candidates by how many additional points they cover
                best_cand = None
                best_score = -1
                cand_list = list(candidates_set)
                if len(cand_list) > 20:
                    cand_list = [cand_list[i] for i in rng.choice(len(cand_list), size=20, replace=False)]
                
                for c_prime in cand_list:
                    ball = neighbors[c_prime]
                    score = np.sum(coverage_count[ball] <= 2)
                    if score > best_score:
                        best_score = score
                        best_cand = c_prime
                
                if best_cand is None:
                    continue
                
                # Apply swap
                old_C = selected_list.copy()
                old_counts = coverage_count.copy()
                
                coverage_count[neighbors[c]] -= 1
                pos = selected_list.index(c)
                selected_list[pos] = best_cand
                selected_set.discard(c)
                selected_set.add(best_cand)
                coverage_count[neighbors[best_cand]] += 1
                
                # Check if swap enables reduction
                new_code, new_counts = reduce_code(selected_list, coverage_count)
                
                if len(new_code) < len(old_C):
                    selected_list = new_code
                    coverage_count = new_counts
                    selected_set = set(new_code)
                    improved = True
                    break
                else:
                    # Revert
                    selected_list = old_C
                    coverage_count = old_counts
                    selected_set = set(old_C)
            
            if time.time() > deadline:
                break
            
            # 3. Two-for-one move: remove two codewords, add one
            if len(selected_list) >= 2:
                two_opt_improved = False
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    
                    i1 = rng.integers(len(selected_list))
                    i2 = rng.integers(len(selected_list))
                    if i1 == i2:
                        continue
                    
                    c1 = selected_list[i1]
                    c2 = selected_list[i2]
                    
                    # Tentatively remove both
                    coverage_count[neighbors[c1]] -= 1
                    coverage_count[neighbors[c2]] -= 1
                    
                    uncovered = np.where(coverage_count == 0)[0]
                    
                    if len(uncovered) == 0:
                        # Both were redundant together
                        selected_set.discard(c1)
                        selected_set.discard(c2)
                        if i1 > i2:
                            i1, i2 = i2, i1
                        selected_list.pop(i1)
                        selected_list.pop(i2)
                        two_opt_improved = True
                        break
                    elif len(uncovered) <= 5:
                        # Find a codeword that covers all uncovered points
                        if len(uncovered) == 1:
                            cand_set = set(neighbors[uncovered[0]].tolist())
                        else:
                            cand_set = set(neighbors[uncovered[0]].tolist())
                            for u in uncovered[1:]:
                                cand_set &= set(neighbors[u].tolist())
                                if not cand_set:
                                    break
                        
                        if cand_set:
                            cand_set.difference_update(selected_set)
                            if cand_set:
                                cand_list = list(cand_set)
                                if len(cand_list) > 15:
                                    cand_list = [cand_list[i] for i in rng.choice(len(cand_list), size=15, replace=False)]
                                
                                best_cand = None
                                best_score = -1
                                for c_prime in cand_list:
                                    ball = neighbors[c_prime]
                                    score = np.sum(coverage_count[ball] <= 2)
                                    if score > best_score:
                                        best_score = score
                                        best_cand = c_prime
                                
                                if best_cand is not None:
                                    pos1 = selected_list.index(c1)
                                    pos2 = selected_list.index(c2)
                                    if pos1 > pos2:
                                        pos1, pos2 = pos2, pos1
                                    selected_list.pop(pos1)
                                    selected_list.pop(pos2)
                                    selected_set.discard(c1)
                                    selected_set.discard(c2)
                                    selected_list.append(best_cand)
                                    selected_set.add(best_cand)
                                    coverage_count[neighbors[best_cand]] += 1
                                    two_opt_improved = True
                                    selected_list, coverage_count = reduce_code(selected_list, coverage_count)
                                    selected_set = set(selected_list)
                                    break
                    coverage_count[neighbors[c1]] += 1
                    coverage_count[neighbors[c2]] += 1
                    
                    if two_opt_improved:
                        break
            
            if improved or two_opt_improved:
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0

    best_code = None
    best_len = float('inf')

    # Run greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 50:
            break
        selected = greedy_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected

    if best_code is None:
        best_code = greedy_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Local Search
    code1 = local_search_v2(best_code, deadline - 5)
    best_code = code1

    # Final Validation
    if not is_valid_code(best_code):
        print("Warning: Local search result invalid, falling back to greedy.")
        fallback_code = greedy_cover(123)
        fallback_code = prune(fallback_code)
        if is_valid_code(fallback_code):
            best_code = fallback_code
        else:
            best_code = list(range(total_points))

    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
