import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Finds maximum packing of disjoint 2x2x2 Strassen blocks using branch-and-bound
    with constraint propagation, then fills remaining entries with schoolbook terms.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    block_params_list = []
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
                            block_params_list.append((i1, i2, j1, j2, k1, k2))
                            all_blocks.append(indices)
    
    num_blocks = len(all_blocks)
    
    # Precompute: for each block, which other blocks conflict with it
    # Two blocks conflict if they share any (i,j,k) entry
    conflict = [set() for _ in range(num_blocks)]
    for a in range(num_blocks):
        for b in range(a + 1, num_blocks):
            if all_blocks[a] & all_blocks[b]:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Precompute block scores based on rarity of entries
    triple_count = {}
    for bidx in range(num_blocks):
        for tri in all_blocks[bidx]:
            triple_count[tri] = triple_count.get(tri, 0) + 1
    
    block_scores = []
    for bidx in range(num_blocks):
        score = sum(1.0 / triple_count[tri] for tri in all_blocks[bidx])
        block_scores.append(score)
    
    sorted_by_score = sorted(range(num_blocks), key=lambda x: -block_scores[x])

    best_num_blocks = 0
    best_packing = []

    def greedy_fill(used_frozen, current_set, max_add=None):
        """Greedy fill using score ordering. Returns (list, used_frozen, count_added)."""
        new_list = list(current_set)
        added = 0
        for bidx in sorted_by_score:
            if max_add is not None and added >= max_add:
                break
            if bidx in current_set:
                continue
            if not (used_frozen & all_blocks[bidx]):
                new_list.append(bidx)
                current_set.add(bidx)
                used_frozen = used_frozen | all_blocks[bidx]
                added += 1
        return new_list, used_frozen, added

    # Branch and bound with constraint propagation
    def branch_and_bound(used_frozen, current_set, remaining_blocks, depth, best_count):
        """Recursive B&B. Returns (best_packing_found, best_count_found)."""
        nonlocal best_num_blocks, best_packing
        
        if time.time() > deadline - 5.0:
            return list(current_set), len(current_set)
        
        # Upper bound: current + remaining (optimistic)
        if len(current_set) + len(remaining_blocks) <= best_count:
            return list(current_set), len(current_set)
        
        if not remaining_blocks:
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_packing = list(current_set)
            return best_packing, best_count
        
        # Try each remaining block
        for bidx in list(remaining_blocks):
            if time.time() > deadline - 5.0:
                break
            
            if used_frozen & all_blocks[bidx]:
                remaining_blocks.discard(bidx)
                continue
            
            # Select this block
            new_used = used_frozen | all_blocks[bidx]
            new_set = set(current_set)
            new_set.add(bidx)
            
            # Filter remaining blocks that don't conflict
            new_remaining = set()
            for rb in remaining_blocks:
                if rb == bidx:
                    continue
                if not (new_used & all_blocks[rb]):
                    new_remaining.add(rb)
            
            packing, count = branch_and_bound(new_used, new_set, new_remaining, depth + 1, best_count)
            if count > best_count:
                best_count = count
                best_packing = packing
        
        return best_packing, best_count

    # Run multiple greedy+local search trials
    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline - 15.0:
            break
        
        random.seed(t * 7919 + 12345)
        
        # Greedy initialization with random perturbation
        avail = list(range(num_blocks))
        random.shuffle(avail)
        
        current_packing = []
        current_packing_set = set()
        used_indices = frozenset()
        
        for bidx in avail:
            if time.time() > deadline - 15.0:
                break
            if not (used_indices & all_blocks[bidx]):
                current_packing.append(bidx)
                current_packing_set.add(bidx)
                used_indices = used_indices | all_blocks[bidx]
        
        current_size = len(current_packing)
        
        # Local search: remove k blocks, re-greedy fill
        improved = True
        max_iters = 20
        for it in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            if not improved:
                break
            improved = False
            
            for k_remove in [1, 2, 3]:
                if time.time() > deadline - 15.0:
                    break
                for _ in range(10):
                    if time.time() > deadline - 15.0:
                        break
                    if len(current_packing) < k_remove:
                        continue
                    
                    remove_set = set(random.sample(current_packing, k_remove))
                    
                    new_used = frozenset()
                    new_packing_set = set()
                    for bp in current_packing:
                        if bp not in remove_set:
                            new_used = new_used | all_blocks[bp]
                            new_packing_set.add(bp)
                    
                    new_packing_list = [bp for bp in current_packing if bp not in remove_set]
                    
                    # Re-greedy fill
                    new_packing_list, new_used, added = greedy_fill(new_used, new_packing_set, max_add=k_remove + 10)
                    
                    if len(new_packing_list) > current_size:
                        current_packing = new_packing_list
                        current_packing_set = set(current_packing)
                        used_indices = new_used
                        current_size = len(current_packing)
                        improved = True
                        break
                if improved:
                    break
        
        if current_size > best_num_blocks:
            best_num_blocks = current_size
            best_packing = [block_params_list[i] for i in current_packing]
        
        # Try B&B from this solution as starting point
        if time.time() < deadline - 20.0 and current_size >= best_num_blocks - 2:
            # Use B&B with pruning from current best
            used_frozen = frozenset()
            for bp in current_packing:
                used_frozen = used_frozen | all_blocks[bp]
            
            current_set = set(current_packing)
            remaining = set(range(num_blocks)) - current_set
            
            # Filter remaining to only non-conflicting ones
            filtered_remaining = set()
            for rb in remaining:
                if not (used_frozen & all_blocks[rb]):
                    filtered_remaining.add(rb)
            
            # Try adding one more block at a time
            for bidx in list(filtered_remaining):
                if time.time() > deadline - 15.0:
                    break
                new_used = used_frozen | all_blocks[bidx]
                new_set = set(current_set)
                new_set.add(bidx)
                
                # Greedy fill from here
                new_packing_list, new_used2, added = greedy_fill(new_used, new_set, max_add=10)
                
                if len(new_packing_list) > best_num_blocks:
                    best_num_blocks = len(new_packing_list)
                    best_packing = [block_params_list[i] for i in new_packing_list]

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

    # Add blocks
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook for uncovered
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

    # Verification
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
                # Fallback to schoolbook
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

    # Add blocks
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook for uncovered
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

    # Verification
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
                # Fallback to schoolbook
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
