import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a refined greedy set-packing approach with multiple heuristics and local search
    to maximize the number of disjoint 2x2x2 Strassen blocks, minimizing the rank.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks_total = len(all_blocks)
    
    # Pre-compute conflict graph for faster local search
    conflict = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    best_num_blocks = 0
    best_packing = []
    best_packing_indices = []

    # Phase 1: Multiple randomized greedy strategies
    max_trials = 5000
    trial = 0
    while trial < max_trials and time.time() < deadline - 30.0:
        trial += 1
        # Use different randomization strategies
        if trial % 5 == 1:
            random.shuffle(all_blocks)
        elif trial % 5 == 2:
            # Sort by some heuristic: prefer blocks that cover "corner" entries
            def block_heuristic(block):
                params, indices = block
                i1, i2, j1, j2, k1, k2 = params
                # Prefer blocks that use extreme indices
                score = 0
                if i1 == 0: score += 1
                if i2 == n-1: score += 1
                if j1 == 0: score += 1
                if j2 == m-1: score += 1
                if k1 == 0: score += 1
                if k2 == p-1: score += 1
                return -score  # Minimize, so prefer higher scores
            sorted_blocks = sorted(all_blocks, key=lambda x: (block_heuristic(x), random.random()))
            all_blocks = sorted_blocks
        elif trial % 5 == 3:
            # Prefer blocks with smaller index ranges (more "compact")
            def compact_heuristic(block):
                params, indices = block
                i1, i2, j1, j2, k1, k2 = params
                return (i2 - i1) + (j2 - j1) + (k2 - k1)
            sorted_blocks = sorted(all_blocks, key=lambda x: (compact_heuristic(x), random.random()))
            all_blocks = sorted_blocks
        elif trial % 5 == 4:
            # Prefer blocks that cover entries with fewer covering blocks (scarce entries)
            entry_count = {}
            for b_params, b_indices in all_blocks:
                for e in b_indices:
                    entry_count[e] = entry_count.get(e, 0) + 1
            def scarcity_heuristic(block):
                params, indices = block
                return sum(entry_count.get(e, 0) for e in indices) / len(indices)
            sorted_blocks = sorted(all_blocks, key=lambda x: (scarcity_heuristic(x), random.random()))
            all_blocks = sorted_blocks
        else:
            # Random order with slight bias
            for idx in range(num_blocks_total - 1, 0, -1):
                swap_idx = random.randint(0, idx)
                all_blocks[idx], all_blocks[swap_idx] = all_blocks[swap_idx], all_blocks[idx]
        
        used_indices = set()
        current_packing = []
        current_packing_indices = []
        
        for idx, (block_params, block_indices) in enumerate(all_blocks):
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                current_packing_indices.append(idx)
                used_indices |= block_indices
        
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
            best_packing_indices = current_packing_indices[:]
            
            if best_num_blocks >= 20:
                break

    # Phase 2: Local search - try to improve by swapping blocks
    if best_num_blocks > 0 and time.time() < deadline - 20.0:
        current_packing = best_packing[:]
        current_packing_indices = best_packing_indices[:]
        used_indices = set()
        for b in current_packing:
            i1, i2, j1, j2, k1, k2 = b
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        used_indices.add((i, j, k))
        
        improved = True
        max_local_iters = 500
        local_iter = 0
        while improved and local_iter < max_local_iters and time.time() < deadline - 20.0:
            improved = False
            local_iter += 1
            
            # Try removing one block and adding two non-conflicting blocks
            for remove_pos in range(len(current_packing)):
                if time.time() > deadline - 20.0:
                    break
                
                removed_idx = current_packing_indices[remove_pos]
                removed_params = current_packing[remove_pos]
                i1, i2, j1, j2, k1, k2 = removed_params
                removed_indices = set()
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            removed_indices.add((i, j, k))
                
                temp_used = used_indices - removed_indices
                temp_packing = current_packing[:remove_pos] + current_packing[remove_pos+1:]
                temp_packing_indices = current_packing_indices[:remove_pos] + current_packing_indices[remove_pos+1:]
                
                # Try to find two blocks that can be added
                candidates = []
                for idx in range(num_blocks_total):
                    if idx not in temp_packing_indices and not conflict[removed_idx][idx]:
                        block_indices = all_blocks[idx][1]
                        if not (block_indices & temp_used):
                            candidates.append(idx)
                
                random.shuffle(candidates)
                
                found_swap = False
                for c1 in candidates[:100]:
                    if found_swap:
                        break
                    b1_indices = all_blocks[c1][1]
                    temp_used_new = temp_used | b1_indices
                    temp_packing_new = temp_packing + [all_blocks[c1][0]]
                    temp_packing_indices_new = temp_packing_indices + [c1]
                    
                    for c2 in candidates[:100]:
                        if c2 == c1:
                            continue
                        b2_indices = all_blocks[c2][1]
                        if not (b2_indices & temp_used_new) and c2 not in temp_packing_indices_new:
                            current_packing = temp_packing_new + [all_blocks[c2][0]]
                            current_packing_indices = temp_packing_indices_new + [c2]
                            used_indices = temp_used_new | b2_indices
                            improved = True
                            found_swap = True
                            
                            if len(current_packing) > best_num_blocks:
                                best_num_blocks = len(current_packing)
                                best_packing = current_packing[:]
                                best_packing_indices = current_packing_indices[:]
                            break
                    if found_swap:
                        break
                if improved:
                    break

    # Phase 3: Try to add more blocks in uncovered regions
    if time.time() < deadline - 10.0:
        used_indices = set()
        for b in best_packing:
            i1, i2, j1, j2, k1, k2 = b
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        used_indices.add((i, j, k))
        
        # Try to add one more block at a time
        added = True
        while added and time.time() < deadline - 10.0:
            added = False
            for idx in range(num_blocks_total):
                if idx in best_packing_indices:
                    continue
                block_indices = all_blocks[idx][1]
                if not (block_indices & used_indices):
                    best_packing.append(all_blocks[idx][0])
                    best_packing_indices.append(idx)
                    used_indices |= block_indices
                    added = True
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
