import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a refined search strategy to maximize the number of disjoint 2x2x2 Strassen blocks.
    The goal is to find the maximum packing of 2x2x2 boxes in a 3x5x7 grid.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    num_blocks_total = len(all_blocks)
    
    # Precompute conflict matrix: conflict[i][j] is True if block i and block j share any index
    # This speeds up local search
    # num_blocks_total is at most 2 * 4 * 6 = 48 blocks. 48x48 matrix is small.
    conflict = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    best_num_blocks = 0
    best_packing = []
    
    # Greedy search with multiple restarts
    for _ in range(500):
        if time.time() > deadline - 20.0:
            break
        random.shuffle(all_blocks)
        used_indices = set()
        current_packing = []
        for block_params, block_indices in all_blocks:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
            if best_num_blocks >= 10: # Stop if we find a very good packing
                break

    # Local search: 1-for-2 swap
    # Try to remove one block and add two others
    current_packing = best_packing[:]
    if current_packing:
        used_indices = set()
        for b in current_packing:
            i1, i2, j1, j2, k1, k2 = b
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        used_indices.add((i, j, k))
        
        # Map block params to index in all_blocks for conflict lookup
        # We need to find the index in all_blocks for each block in current_packing
        # Since all_blocks is shuffled in the loop, we can't rely on fixed indices if we don't save them.
        # Let's create a stable list of blocks and indices.
        
        # Re-initialize with a stable list
        stable_blocks = [b[0] for b in all_blocks]
        stable_indices = [b[1] for b in all_blocks]
        
        # Re-run best greedy on stable blocks to get consistent indices
        # Actually, let's just use the set of block params to identify them.
        # For local search, we operate on the list of block params.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to remove one block
            for idx_remove in range(len(current_packing)):
                if time.time() > deadline - 5.0:
                    break
                removed_block = current_packing[idx_remove]
                
                # Find indices of removed block
                i1, i2, j1, j2, k1, k2 = removed_block
                removed_indices = set()
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            removed_indices.add((i, j, k))
                
                # Temporarily remove
                temp_used = used_indices - removed_indices
                temp_packing = current_packing[:idx_remove] + current_packing[idx_remove+1:]
                temp_packing_set = set(temp_packing)
                
                # Find all blocks that can be added to temp_used
                # A block can be added if it doesn't conflict with temp_used
                # And is not already in temp_packing
                candidates = []
                for b in stable_blocks:
                    if b in temp_packing_set:
                        continue
                    # Check if b conflicts with temp_used
                    # We need the indices of b. We can look it up or precompute.
                    # Let's precompute a dict from params to indices
                    # This is inefficient if done inside loop.
                    # Let's assume we have a map.
                    pass
                
                # Optimization: Precompute map from params to indices
                # This should be done once.
                # Let's restructure.

    # Let's restart the local search part with better data structures
    
    # Precompute params to indices
    params_to_indices = {b[0]: b[1] for b in all_blocks}
    
    # Re-initialize best packing from greedy
    # (Assuming best_packing is already found)
    
    if best_packing:
        used_indices = set()
        for b in best_packing:
            used_indices |= params_to_indices[b]
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for idx_remove in range(len(best_packing)):
                if time.time() > deadline - 5.0:
                    break
                removed_block = best_packing[idx_remove]
                removed_indices = params_to_indices[removed_block]
                
                temp_used = used_indices - removed_indices
                temp_packing = best_packing[:idx_remove] + best_packing[idx_remove+1:]
                temp_packing_set = set(temp_packing)
                
                # Find candidates that are disjoint from temp_used
                candidates = []
                for b in stable_blocks:
                    if b in temp_packing_set:
                        continue
                    if not (params_to_indices[b] & temp_used):
                        candidates.append(b)
                
                # Try to find two disjoint candidates
                # To speed up, we can just pick two random ones from candidates and check
                # Or iterate. Candidates list is small.
                random.shuffle(candidates)
                found_swap = False
                for c1 in candidates:
                    if found_swap: break
                    idx1 = params_to_indices[c1]
                    temp_used_1 = temp_used | idx1
                    for c2 in candidates:
                        if c2 == c1: continue
                        idx2 = params_to_indices[c2]
                        if not (idx2 & temp_used_1):
                            # Found a valid swap: remove 1, add 2
                            best_packing = temp_packing + [c1, c2]
                            used_indices = temp_used_1 | idx2
                            improved = True
                            found_swap = True
                            break
                if improved:
                    break
                    # break outer loop? No, just break inner and continue while loop
                    # Actually, if we improved, we should restart the scan or continue
                    # Breaking the for loop and letting while loop restart is safe

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
