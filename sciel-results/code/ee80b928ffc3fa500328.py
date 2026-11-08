import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an improved local search for maximum packing of disjoint 2x2x2 Strassen blocks.
    Combines greedy initialization with score-based local search (remove k, re-greedy fill).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices = indices | frozenset([(i, j, k)])
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    num_blocks_total = len(all_blocks)
    
    # Precompute triple frequencies for scoring
    triple_count = {}
    for bidx in range(num_blocks_total):
        for tri in all_blocks[bidx][1]:
            triple_count[tri] = triple_count.get(tri, 0) + 1
            
    # Score blocks: sum of inverse frequencies (rarer triples are more valuable)
    block_scores = []
    for bidx in range(num_blocks_total):
        score = sum(1.0 / triple_count[tri] for tri in all_blocks[bidx][1])
        block_scores.append(score)
    
    # Pre-sort block indices by score descending for greedy filling
    sorted_by_score = sorted(range(num_blocks_total), key=lambda x: -block_scores[x])

    best_num_blocks = 0
    best_packing = []

    # Helper to greedily add blocks to a packing
    def greedy_pack(current_packing, used_indices, max_add=None):
        added = 0
        new_packing = current_packing[:]
        for bidx in sorted_by_score:
            if max_add is not None and added >= max_add:
                break
            if bidx in current_packing:
                continue
            idx = all_blocks[bidx][1]
            if not (used_indices & idx):
                new_packing.append(bidx)
                used_indices = used_indices | idx
                added += 1
        return new_packing, used_indices, added

    # Local Search: Remove k blocks, re-greedy fill
    # We focus on moves that increase the count
    num_trials = 500
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(t + 54321)
        
        # Initialize with a random subset or empty, then greedy fill
        # Start with a random permutation of all blocks
        avail = list(range(num_blocks_total))
        random.shuffle(avail)
        
        current_packing = []
        used_indices = set()
        
        # Greedy fill
        for bidx in avail:
            if time.time() > deadline - 10.0:
                break
            idx = all_blocks[bidx][1]
            if not (used_indices & idx):
                current_packing.append(bidx)
                used_indices |= idx
        
        current_size = len(current_packing)
        
        # Local Search: Remove k blocks, re-greedy fill
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try removing 1 or 2 random blocks and re-packing
            for k_remove in [1, 2]:
                if time.time() > deadline - 10.0:
                    break
                
                # Try a few random removals
                for _ in range(10):
                    if time.time() > deadline - 10.0:
                        break
                    
                    if len(current_packing) < k_remove:
                        continue
                    
                    remove_set = random.sample(current_packing, k_remove)
                    remove_set_set = set(remove_set)
                    
                    new_used = set()
                    for bp in current_packing:
                        if bp not in remove_set_set:
                            new_used |= all_blocks[bp][1]
                            
                    new_packing = [bp for bp in current_packing if bp not in remove_set_set]
                    
                    # Re-greedy fill
                    # We can try to add any block not in new_packing
                    # To speed up, we just iterate through all blocks sorted by score
                    # But we must be careful not to re-add the ones we just removed if they are the best fit?
                    # Actually, re-adding is fine, it's just a local search move.
                    
                    # Optimization: Only consider blocks that conflict with the removed set or were previously blocked?
                    # Simpler: Just re-greedy from scratch on the available space.
                    # But that's O(N^2). Let's just try to add blocks that were not in the original packing
                    # or were removed.
                    
                    # Actually, the best way is to try to add blocks that fit in the new space.
                    # We'll iterate through all blocks, prioritized by score.
                    
                    added_count = 0
                    # We want to see if we can get size > current_size.
                    # If we remove k, we need to add > k to improve.
                    # Let's try to add up to k+5 blocks to see if we can improve.
                    max_add = k_remove + 5
                    
                    new_packing, new_used, added = greedy_pack(new_packing, new_used, max_add)
                    
                    if len(new_packing) > current_size:
                        current_packing = new_packing
                        used_indices = new_used
                        current_size = len(current_packing)
                        improved = True
                        break # Restart the k_remove loop with new state
                    
                    if improved:
                        break

        if current_size > best_num_blocks:
            best_num_blocks = current_size
            best_packing = current_packing[:]

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
    for block_idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[block_idx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for tri in all_blocks[block_idx][1]:
            covered.add(tri)

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

    # Verification Step
    if rank > 0:
        try:
            U_np = np.array(U, dtype=np.int64)
            V_np = np.array(V, dtype=np.int64)
            W_np = np.array(W, dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        prod = U_np[u_idx] * V_np[v_idx]
                        val = np.dot(prod, W_np[w_idx])
                        
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                        break
                        
            if not valid:
                # Fall back to schoolbook
                U_cols = []
                V_cols = []
                W_cols = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
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
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
