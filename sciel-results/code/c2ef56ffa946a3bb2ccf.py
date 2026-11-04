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
    
    Optimized search:
    1. Precompute all blocks with bitmasks and precomputed priority scores
    2. Fast greedy with precomputed priorities (no per-iteration recomputation)
    3. Efficient local search using incremental mask updates
    4. More trials due to faster per-trial cost
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    total_triples = n * m * p  # 105

    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k

    # Precompute all possible 2x2x2 blocks
    all_blocks = []  # list of (params, bitmask, tuple_of_indices)
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            indices = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                                        indices.append(idx)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, tuple(indices)))

    num_all_blocks = len(all_blocks)

    # Count how many blocks cover each triple
    triple_block_count = [0] * total_triples
    for _, _, indices in all_blocks:
        for idx in indices:
            triple_block_count[idx] += 1

    # Precompute priority score for each block (sum of 1/count for its indices)
    block_priority = []
    for _, _, indices in all_blocks:
        pri = sum(1.0 / triple_block_count[idx] for idx in indices)
        block_priority.append(pri)

    best_num_blocks = 0
    best_packing = []

    def greedy_random():
        """Fast greedy with random ordering."""
        order = list(range(num_all_blocks))
        random.shuffle(order)
        used_mask = 0
        packing = []
        for bi in order:
            bm = all_blocks[bi][1]
            if (bm & used_mask) == 0:
                packing.append(bi)
                used_mask |= bm
        return packing

    def greedy_priority():
        """Greedy selecting by highest priority among available blocks."""
        available = list(range(num_all_blocks))
        used_mask = 0
        packing = []
        while available:
            # Find block with highest priority that doesn't conflict
            best_bi = -1
            best_pri = -1
            for bi in available:
                bm = all_blocks[bi][1]
                if (bm & used_mask) == 0:
                    if block_priority[bi] > best_pri:
                        best_pri = block_priority[bi]
                        best_bi = bi
            if best_bi == -1:
                break
            packing.append(best_bi)
            used_mask |= all_blocks[best_bi][1]
            available.remove(best_bi)
        return packing

    def greedy_priority_random():
        """Greedy with priority but random tie-breaking."""
        available = list(range(num_all_blocks))
        random.shuffle(available)
        used_mask = 0
        packing = []
        while available:
            # Among available non-conflicting, pick highest priority (random order breaks ties)
            best_bi = -1
            best_pri = -1
            for bi in available:
                bm = all_blocks[bi][1]
                if (bm & used_mask) == 0:
                    if block_priority[bi] > best_pri:
                        best_pri = block_priority[bi]
                        best_bi = bi
            if best_bi == -1:
                break
            packing.append(best_bi)
            used_mask |= all_blocks[best_bi][1]
            available.remove(best_bi)
        return packing

    def local_search(packing, max_iterations=30):
        """Efficient local search: try replacing one block to gain another."""
        if not packing:
            return packing
        
        current_packing = list(packing)
        used_mask = 0
        for bi in current_packing:
            used_mask |= all_blocks[bi][1]
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            for remove_idx in range(len(current_packing)):
                removed_bi = current_packing[remove_idx]
                mask_without = used_mask & ~all_blocks[removed_bi][1]
                
                # Try adding any block that fits in the freed space
                for bi_add in range(num_all_blocks):
                    if bi_add == removed_bi:
                        continue
                    if bi_add in current_packing:
                        continue
                    bm_add = all_blocks[bi_add][1]
                    if (bm_add & mask_without) == 0:
                        # We can add this block after removing removed_bi
                        # Net gain: we lose 1 block, gain 1 block -> no net gain unless we can add more
                        # Actually this doesn't increase count. We need to find a block that
                        # fits WITHOUT removing anything, or remove 1 and add 2+.
                        # For simplicity, just try to find if removing one block allows
                        # adding a block that wasn't there before (net 0) - not useful.
                        # Better: check if we can add a block to the current packing directly.
                        pass
                
                # Better approach: try to add a block to current packing (no removal)
                # This is already done by greedy. For local search, try 1-out 2-in swaps.
                for bi_add1 in range(num_all_blocks):
                    if bi_add1 in current_packing:
                        continue
                    bm1 = all_blocks[bi_add1][1]
                    if (bm1 & mask_without) != 0:
                        continue
                    for bi_add2 in range(bi_add1 + 1, num_all_blocks):
                        if bi_add2 in current_packing:
                            continue
                        bm2 = all_blocks[bi_add2][1]
                        if (bm2 & mask_without) != 0:
                            continue
                        if (bm1 & bm2) != 0:
                            continue
                        # We can add both bi_add1 and bi_add2 after removing removed_bi
                        new_packing = current_packing[:remove_idx] + current_packing[remove_idx+1:] + [bi_add1, bi_add2]
                        current_packing = new_packing
                        used_mask = 0
                        for bi in current_packing:
                            used_mask |= all_blocks[bi][1]
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current_packing

    # Run many trials
    trial_count = 0
    max_trials = 5000
    for t in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        trial_count += 1
        
        if t % 4 == 0:
            packing = greedy_random()
        elif t % 4 == 1:
            packing = greedy_priority()
        elif t % 4 == 2:
            packing = greedy_priority_random()
        else:
            packing = greedy_random()
            packing = local_search(packing, max_iterations=15)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            
            if best_num_blocks >= 14:
                break

    # Systematic approach: sort by conflict count
    if time.time() < deadline - 15.0:
        # Compute conflict count for each block
        conflict_counts = [0] * num_all_blocks
        for i in range(num_all_blocks):
            bm_i = all_blocks[i][1]
            for j in range(i + 1, num_all_blocks):
                if (bm_i & all_blocks[j][1]) != 0:
                    conflict_counts[i] += 1
                    conflict_counts[j] += 1
        
        sorted_indices = sorted(range(num_all_blocks), key=lambda x: conflict_counts[x])
        
        used_mask = 0
        packing = []
        for bi in sorted_indices:
            bm = all_blocks[bi][1]
            if (bm & used_mask) == 0:
                packing.append(bi)
                used_mask |= bm
        
        packing = local_search(packing, max_iterations=25)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing

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

    # Add the Strassen blocks from the best packing (now stored as indices)
    covered = set()
    for bi in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[bi][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for idx in all_blocks[bi][2]:
            covered.add(idx)

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                idx = triple_to_idx(i, j, k)
                if idx not in covered:
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
