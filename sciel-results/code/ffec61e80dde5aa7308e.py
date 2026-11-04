import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    
    Key improvements:
    1. Precompute all blocks once with their index sets
    2. Use bitmasks for fast disjointness checking
    3. Much faster local search using precomputed block masks
    4. Multiple random restarts with different random orderings
    5. Try both pure random shuffle and priority-based ordering
    6. Fast 1-swap and 2-swap local search
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Total number of (i,j,k) triples
    total_triples = n * m * p  # 105

    # Precompute all possible 2x2x2 blocks with bitmask representation
    all_blocks = []  # list of (params, bitmask, index_set)
    
    # Index of each (i,j,k) triple to a unique bit position
    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, indices))
    
    num_all_blocks = len(all_blocks)
    
    # Precompute block bitmasks as a list for fast access
    block_masks = [bm for _, bm, _ in all_blocks]
    block_params_list = [params for params, _, _ in all_blocks]
    
    # Count how many blocks cover each triple (for rarity heuristic)
    triple_block_count = [0] * total_triples
    for _, _, indices in all_blocks:
        for i, j, k in indices:
            triple_block_count[triple_to_idx(i, j, k)] += 1
    
    # Precompute for each block, which other blocks conflict with it
    # conflict_mask[b] = bitmask of blocks that conflict with block b
    # This is too large (2^num_blocks), so instead store list of conflicting block indices
    block_conflicts = []
    for idx in range(num_all_blocks):
        bm = block_masks[idx]
        conflicts = []
        for other in range(num_all_blocks):
            if other != idx and (bm & block_masks[other]) != 0:
                conflicts.append(other)
        block_conflicts.append(conflicts)

    best_num_blocks = 0
    best_packing = []

    def greedy_random():
        """Greedy selection with random ordering."""
        shuffled = list(range(num_all_blocks))
        random.shuffle(shuffled)
        used_mask = 0
        packing = []
        for bidx in shuffled:
            bm = block_masks[bidx]
            if (bm & used_mask) == 0:
                packing.append(bidx)
                used_mask |= bm
        return packing

    def greedy_priority():
        """Greedy selection prioritizing blocks that cover rare indices."""
        shuffled = list(range(num_all_blocks))
        random.shuffle(shuffled)
        used_mask = 0
        packing = []
        for bidx in shuffled:
            bm = block_masks[bidx]
            if (bm & used_mask) == 0:
                # Compute priority based on uncovered rare indices
                priority = 0
                for i, j, k in all_blocks[bidx][2]:
                    tidx = triple_to_idx(i, j, k)
                    if (1 << tidx) & used_mask == 0:
                        priority += 1.0 / triple_block_count[tidx]
                if priority > 0:
                    packing.append(bidx)
                    used_mask |= bm
        return packing

    def fast_local_search(packing, max_iterations=20):
        """Fast local search using precomputed block masks."""
        if not packing:
            return packing
        
        current = list(packing)
        # Build used mask
        used_mask = 0
        for bidx in current:
            used_mask |= block_masks[bidx]
        
        # Build set for O(1) lookup
        current_set = set(current)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            improved = False
            
            # Try 1-swap: remove one block, try to add a different one
            for pos in range(len(current)):
                removed_bidx = current[pos]
                mask_without = used_mask & ~block_masks[removed_bidx]
                
                # Try to find a block that fits in the freed space
                # Only consider blocks that were conflicting with removed_bidx
                for cand in block_conflicts[removed_bidx]:
                    if cand not in current_set:
                        if (block_masks[cand] & mask_without) == 0:
                            # Found a valid swap
                            current[pos] = cand
                            current_set.remove(removed_bidx)
                            current_set.add(cand)
                            used_mask = mask_without | block_masks[cand]
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try 2-swap: remove two blocks, add two different ones
                found = False
                for r1 in range(len(current)):
                    if time.time() > deadline - 3.0:
                        break
                    for r2 in range(r1 + 1, len(current)):
                        b1 = current[r1]
                        b2 = current[r2]
                        mask_without = used_mask & ~block_masks[b1] & ~block_masks[b2]
                        
                        # Try to find two blocks that fit
                        added = []
                        for cand in block_conflicts[b1]:
                            if cand not in current_set and cand != b2:
                                if (block_masks[cand] & mask_without) == 0:
                                    added.append(cand)
                                    mask_without |= block_masks[cand]
                                    if len(added) == 2:
                                        break
                        
                        if len(added) == 2:
                            # Verify second block also works
                            # Re-check
                            mask_check = used_mask & ~block_masks[b1] & ~block_masks[b2]
                            valid = True
                            for cand in added:
                                if (block_masks[cand] & mask_check) != 0:
                                    valid = False
                                    break
                                mask_check |= block_masks[cand]
                            
                            if valid:
                                # Apply swap
                                new_current = []
                                new_set = set()
                                for idx, bidx in enumerate(current):
                                    if idx != r1 and idx != r2:
                                        new_current.append(bidx)
                                        new_set.add(bidx)
                                new_current.extend(added)
                                new_set.update(added)
                                current = new_current
                                current_set = new_set
                                used_mask = 0
                                for bidx in current:
                                    used_mask |= block_masks[bidx]
                                improved = True
                                found = True
                                break
                    if found:
                        break
            
            if not improved:
                break
        
        return current

    # Main search loop
    num_trials = 20000
    trial_count = 0
    for t in range(num_trials):
        if time.time() > deadline - 3.0:
            break
        trial_count += 1
        
        if t % 4 == 0:
            packing = greedy_random()
        elif t % 4 == 1:
            packing = greedy_priority()
        elif t % 4 == 2:
            packing = greedy_random()
            packing = fast_local_search(packing, max_iterations=15)
        else:
            packing = greedy_priority()
            packing = fast_local_search(packing, max_iterations=15)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = list(packing)
            
            # Early exit if we hit a good number
            if best_num_blocks >= 12:
                break

    # Also try a more systematic approach: sort by conflict count
    if time.time() < deadline - 5.0:
        # Sort blocks by number of conflicts (fewer conflicts first)
        block_conflict_counts = [(len(block_conflicts[idx]), idx) for idx in range(num_all_blocks)]
        block_conflict_counts.sort()
        sorted_indices = [idx for _, idx in block_conflict_counts]
        
        # Greedy on sorted order
        used_mask = 0
        packing = []
        for idx in sorted_indices:
            if (block_masks[idx] & used_mask) == 0:
                packing.append(idx)
                used_mask |= block_masks[idx]
        
        # Apply local search
        packing = fast_local_search(packing, max_iterations=30)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = list(packing)

    # Additional: try random restarts focused on improving the best solution
    if time.time() < deadline - 5.0 and best_packing:
        for _ in range(100):
            if time.time() > deadline - 3.0:
                break
            # Start from best and perturb
            perturbed = list(best_packing)
            # Remove 1-2 random blocks
            for _ in range(random.randint(1, 2)):
                if perturbed:
                    rem_idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(rem_idx)
            # Add local search
            perturbed = fast_local_search(perturbed, max_iterations=10)
            if len(perturbed) > best_num_blocks:
                best_num_blocks = len(perturbed)
                best_packing = list(perturbed)
                if best_num_blocks >= 12:
                    break

    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for bidx in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params_list[bidx]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
