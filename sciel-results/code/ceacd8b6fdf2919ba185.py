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
    Theoretical max is 9 blocks, which would yield rank 96 (105 - 9).
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

    # Greedy search for maximum disjoint packing
    # We try multiple random orderings and local improvements
    best_num_blocks = 0
    best_packing = []
    
    # Initial greedy pass
    for _ in range(200):
        if time.time() > deadline - 10.0:
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
            best_packing = current_packing
            if best_num_blocks >= 9:
                break

    # Local search to try to improve the packing
    if best_num_blocks < 9 and time.time() < deadline - 5.0:
        current_packing = best_packing[:]
        used_indices = set()
        for b in current_packing:
            # Re-calculate indices for the current best
            i1, i2, j1, j2, k1, k2 = b
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        used_indices.add((i, j, k))
        
        # Map block params to indices for quick lookup
        block_idx_map = {b: i for i, (b, _) in enumerate(all_blocks)}
        # Actually, we just need to know which blocks conflict with which
        # For local search, we try to swap blocks
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to remove one block and add two others
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
                
                # Try to add any two disjoint blocks from the remaining pool
                # This is expensive, so we just try random additions
                available_blocks = [b for b in all_blocks if b[0] not in temp_packing]
                random.shuffle(available_blocks)
                
                added_any = False
                for block_params, block_indices in available_blocks[:50]: # Limit search
                    if not (block_indices & temp_used):
                        # Can we add a second one?
                        temp_used_new = temp_used | block_indices
                        temp_packing_new = temp_packing + [block_params]
                        for block_params2, block_indices2 in available_blocks[:50]:
                            if block_params2 == block_params:
                                continue
                            if not (block_indices2 & temp_used_new) and block_params2 not in temp_packing_new:
                                # We found a swap that increases count by 1 (remove 1, add 2)
                                current_packing = temp_packing_new + [block_params2]
                                used_indices = temp_used_new | block_indices2
                                improved = True
                                if len(current_packing) > best_num_blocks:
                                    best_num_blocks = len(current_packing)
                                    best_packing = current_packing
                                break
                        if improved:
                            break
                if improved:
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
