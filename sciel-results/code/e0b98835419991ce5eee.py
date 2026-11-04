import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a search for disjoint Strassen blocks to minimize rank.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

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

    # Generate all possible 2x2x2 blocks
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
    
    # Sort blocks by some heuristic? e.g. those covering more "rare" indices?
    # Or just random search with backtracking for max packing.
    
    best_packing = []
    
    # Try to find a packing of size 8
    # If found, use it. Else, find best possible (likely 7).
    
    # Backtracking search for max packing
    # This might be slow, so we limit it.
    
    def search_packing(used_indices, current_packing, remaining_blocks, target_size):
        if len(current_packing) == target_size:
            return current_packing[:]
        if not remaining_blocks:
            return None
        if time.time() > deadline - 10.0:
            return None
            
        # Heuristic: try blocks that are "central" or cover more used space?
        # Just iterate
        for idx, (params, indices) in enumerate(remaining_blocks):
            if not (indices & used_indices):
                new_used = used_indices | indices
                new_packing = current_packing + [params]
                result = search_packing(new_used, new_packing, remaining_blocks[idx+1:], target_size)
                if result:
                    return result
        return None

    # Try to find 8 blocks
    # Randomize order to escape local minima
    found_8 = False
    for trial in range(50):
        if time.time() > deadline - 20.0:
            break
        random.shuffle(all_blocks)
        # Use a subset for search to speed up?
        # The search above is exponential. 50 trials of exponential search might be too slow.
        # Let's stick to greedy but with more trials and better initialization.
        
        # Actually, let's just do a very thorough greedy search.
        # And also try to swap blocks to improve packing.
        pass

    # Fallback: Greedy search for best packing (likely 7)
    best_num_blocks = 0
    best_packing = []
    
    num_trials = 2000
    for t in range(num_trials):
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
            if best_num_blocks == 8:
                break

    # If we found 8 blocks, great. If 7, also good.
    # If we can find a better remainder decomposition, we use it.
    
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
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # For the remainder, we use schoolbook.
    # We could try to optimize the remainder, but it's complex.
    # If best_num_blocks is 7, rank is 98.
    # If we can find 8, rank is 97.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
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
