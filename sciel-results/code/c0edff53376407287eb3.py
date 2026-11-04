import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a maximum packing of disjoint 2x2x2 Strassen blocks for 3x5x7 matrix multiplication,
    then adds schoolbook terms for the remaining (i,j,k) indices.
    Uses randomized greedy with multiple restarts + enhanced local search (add/swap/move operations).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Precompute all possible 2x2x2 blocks
    all_blocks = []
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
                                        idx = i * m * p + j * p + k
                                        bitmask |= (1 << idx)
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, indices))

    num_blocks_total = len(all_blocks)
    
    # Precompute conflict masks for faster checking? 
    # Or just use bitmask AND.
    
    best_num_blocks = 0
    best_packing = []

    def greedy_random():
        shuffled = list(range(num_blocks_total))
        random.shuffle(shuffled)
        used_mask = 0
        packing = []
        for bi in shuffled:
            bm = all_blocks[bi][1]
            if (bm & used_mask) == 0:
                packing.append(all_blocks[bi][0])
                used_mask |= bm
        return packing

    def local_search(packing, max_iters=100):
        if not packing:
            return packing
        
        current_packing = list(packing)
        current_set = set(current_packing)
        used_mask = 0
        for params in current_packing:
            i1, i2, j1, j2, k1, k2 = params
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        idx = i * m * p + j * p + k
                        used_mask |= (1 << idx)
        
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline - 5.0:
            improved = False
            it += 1
            
            # 1. Try adding any block
            for bi in range(num_blocks_total):
                params = all_blocks[bi][0]
                if params in current_set:
                    continue
                bm = all_blocks[bi][1]
                if (bm & used_mask) == 0:
                    current_packing.append(params)
                    current_set.add(params)
                    used_mask |= bm
                    improved = True
                    break
            
            if not improved:
                # 2. Try swap: remove 1, add 2
                for ri in range(len(current_packing)):
                    if time.time() > deadline - 5.0:
                        break
                    remove_params = current_packing[ri]
                    remove_bm = 0
                    for i1, i2, j1, j2, k1, k2 in [remove_params]:
                        for i in (i1, i2):
                            for j in (j1, j2):
                                for k in (k1, k2):
                                    idx = i * m * p + j * p + k
                                    remove_bm |= (1 << idx)
                    
                    trial_used = used_mask & ~remove_bm
                    trial_set = current_set - {remove_params}
                    trial_packing = [x for x in current_packing if x != remove_params]
                    
                    added = 0
                    for bi in range(num_blocks_total):
                        if bi in trial_set:
                            continue
                        bm = all_blocks[bi][1]
                        if (bm & trial_used) == 0:
                            trial_packing.append(all_blocks[bi][0])
                            trial_set.add(bi)
                            trial_used |= bm
                            added += 1
                            if added >= 2:
                                break
                    
                    if len(trial_packing) > len(current_packing):
                        current_packing = trial_packing
                        current_set = trial_set
                        used_mask = trial_used
                        improved = True
                        break
            
            if not improved:
                # 3. Try swap: remove 2, add 3
                for r1 in range(len(current_packing)):
                    if time.time() > deadline - 5.0:
                        break
                    for r2 in range(r1 + 1, len(current_packing)):
                        if time.time() > deadline - 5.0:
                            break
                        remove1 = current_packing[r1]
                        remove2 = current_packing[r2]
                        
                        remove_bm = 0
                        for params in [remove1, remove2]:
                            i1, i2, j1, j2, k1, k2 = params
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = i * m * p + j * p + k
                                        remove_bm |= (1 << idx)
                        
                        trial_used = used_mask & ~remove_bm
                        trial_set = current_set - {remove1, remove2}
                        trial_packing = [x for x in current_packing if x not in (remove1, remove2)]
                        
                        added = 0
                        for bi in range(num_blocks_total):
                            if bi in trial_set:
                                continue
                            bm = all_blocks[bi][1]
                            if (bm & trial_used) == 0:
                                trial_packing.append(all_blocks[bi][0])
                                trial_set.add(bi)
                                trial_used |= bm
                                added += 1
                                if added >= 3:
                                    break
                        
                        if len(trial_packing) > len(current_packing):
                            current_packing = trial_packing
                            current_set = trial_set
                            used_mask = trial_used
                            improved = True
                            break
                    if improved:
                        break
        
        return current_packing

    # Run multiple restarts
    num_restarts = 500
    for t in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        packing = greedy_random()
        packing = local_search(packing, max_iters=50)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            if best_num_blocks >= 10: # Hope for improvement
                break

    # Construct U, V, W
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
