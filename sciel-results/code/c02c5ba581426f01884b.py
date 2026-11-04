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
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    # A block is defined by two i's, two j's, two k's
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # The set of indices covered by this block
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    # Greedy search with scoring heuristic + local search swaps
    # Score each block by how many future blocks it conflicts with (prefer less-conflicting)
    # Then try local search: swap one block for two that fit in the freed space
    
    # Pre-compute conflict counts for each block
    num_blocks = len(all_blocks)
    conflict_count = [0] * num_blocks
    for a in range(num_blocks):
        for b in range(a + 1, num_blocks):
            if all_blocks[a][1] & all_blocks[b][1]:
                conflict_count[a] += 1
                conflict_count[b] += 1
    
    best_num_blocks = 0
    best_packing = []
    
    def greedy_with_scores(order=None):
        """Greedy: pick blocks in given order (or by conflict score), skip conflicts."""
        used_indices = set()
        current_packing = []
        if order is None:
            # Sort by conflict count ascending (prefer blocks that conflict with fewer others)
            indices = sorted(range(num_blocks), key=lambda x: (conflict_count[x], random.random()))
        else:
            indices = order
        for idx in indices:
            block_params, block_indices = all_blocks[idx]
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        return current_packing
    
    def local_search(packing):
        """Try to improve packing by swapping one block for two others."""
        if len(packing) == 0:
            return packing
        best = list(packing)
        best_size = len(best)
        
        # Build set of used indices
        used = set()
        for bp in best:
            for i in (bp[0], bp[1]):
                for j in (bp[2], bp[3]):
                    for k in (bp[4], bp[5]):
                        used.add((i, j, k))
        
        # For each block in packing, try removing it and fitting two new blocks
        for remove_idx in range(len(best)):
            if time.time() > deadline - 3.0:
                break
            # Remove this block
            removed_block = best[remove_idx]
            removed_indices = set()
            for i in (removed_block[0], removed_block[1]):
                for j in (removed_block[2], removed_block[3]):
                    for k in (removed_block[4], removed_block[5]):
                        removed_indices.add((i, j, k))
            
            new_used = used - removed_indices
            candidate = [b for i, b in enumerate(best) if i != remove_idx]
            
            # Try to add two blocks that fit in the freed space
            added = 0
            for a in range(num_blocks):
                if added >= 2:
                    break
                bp_a, bi_a = all_blocks[a]
                # Must not conflict with new_used
                if bi_a & new_used:
                    continue
                # Must be in the freed region (overlap with removed_indices)
                if not (bi_a & removed_indices):
                    continue
                # Check not already in candidate
                if bp_a in candidate:
                    continue
                candidate.append(bp_a)
                new_used |= bi_a
                added += 1
                
                # Try to add a second
                for b in range(a + 1, num_blocks):
                    if added >= 2:
                        break
                    bp_b, bi_b = all_blocks[b]
                    if bi_b & new_used:
                        continue
                    if not (bi_b & removed_indices):
                        continue
                    if bp_b in candidate:
                        continue
                    candidate.append(bp_b)
                    new_used |= bi_b
                    added += 1
                    break
            
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
                # Rebuild used set
                used = set()
                for bp in best:
                    for i in (bp[0], bp[1]):
                        for j in (bp[2], bp[3]):
                            for k in (bp[4], bp[5]):
                                used.add((i, j, k))
        
        return best
    
    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
        
        # Mix of strategies: some random, some sorted by conflict
        if t % 3 == 0:
            order = list(range(num_blocks))
            random.shuffle(order)
        elif t % 3 == 1:
            order = None  # Use conflict-based sorting
        else:
            # Hybrid: sort by conflict but with some randomization
            order = sorted(range(num_blocks), key=lambda x: (conflict_count[x] + random.random() * 5, random.random()))
        
        packing = greedy_with_scores(order)
        
        # Local search improvement
        packing = local_search(packing)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            
            # Early exit if we hit the theoretical max
            if best_num_blocks >= 13:
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
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    # Try to merge nearby schoolbook terms into additional Strassen blocks
    # Look for 2x2x2 subgrids that are all schoolbook terms and can be converted
    # This is a second-pass optimization
    remaining_schoolbook = [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
    
    # Check if any 2x2x2 block can be formed from remaining schoolbook terms
    # and would save rank (7 vs 8)
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            block_triples = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        block_triples.add((i, j, k))
                            # All 8 must be schoolbook (uncovered)
                            if block_triples.issubset(remaining_schoolbook):
                                # Remove these 8 schoolbook terms and add Strassen block
                                # Find and remove the corresponding columns
                                # This is complex; skip for now as the initial packing should handle this
                                pass

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
