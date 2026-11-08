import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    block_indices_list = []
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset(
                                (i, j, k)
                                for i in (i1, i2)
                                for j in (j1, j2)
                                for k in (k1, k2)
                            )
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
                            block_indices_list.append(indices)

    total_blocks = len(all_blocks)
    
    # Precompute conflicts
    # conflict[i] is a set of block indices that conflict with block i
    conflicts = [set() for _ in range(total_blocks)]
    for i in range(total_blocks):
        for j in range(i + 1, total_blocks):
            if block_indices_list[i] & block_indices_list[j]:
                conflicts[i].add(j)
                conflicts[j].add(i)

    best_packing = []
    best_size = 0
    
    random.seed(42)
    
    # Local Search with Random Restarts
    for trial in range(500):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with greedy
        current_packing = set()
        available = list(range(total_blocks))
        random.shuffle(available)
        
        for idx in available:
            if not (current_packing & conflicts[idx]):
                current_packing.add(idx)
        
        current_size = len(current_packing)
        
        # Local Improvement
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try to add a block
            # Pick a random available block not in current_packing
            # This is O(N) to find one.
            # Better: maintain a list of candidates?
            # For N=630, simple loop is fine.
            
            # Try 1-for-0 (Add)
            for _ in range(10): # Try 10 random additions
                idx = random.randint(0, total_blocks - 1)
                if idx in current_packing:
                    continue
                if not (current_packing & conflicts[idx]):
                    current_packing.add(idx)
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try 2-for-1 (Swap)
            # Remove one block, try to add two
            if current_size > 0:
                remove_idx = random.choice(list(current_packing))
                # Tentatively remove
                current_packing.remove(remove_idx)
                current_size -= 1
                
                added = 0
                # Try to add 2 blocks that don't conflict with the new set
                candidates = [i for i in range(total_blocks) if i not in current_packing and i != remove_idx]
                random.shuffle(candidates)
                
                temp_packing = set(current_packing)
                for idx in candidates:
                    if added == 2:
                        break
                    if not (temp_packing & conflicts[idx]):
                        temp_packing.add(idx)
                        added += 1
                
                if added == 2:
                    current_packing = temp_packing
                    current_size += 1 # Net gain 1
                    improved = True
                else:
                    # Revert
                    current_packing.add(remove_idx)
                    current_size += 1
            
            # Try 3-for-2 (Swap)
            if current_size > 1 and not improved:
                # Remove 2, try to add 3
                remove_indices = random.sample(list(current_packing), 2)
                for r in remove_indices:
                    current_packing.remove(r)
                current_size -= 2
                
                temp_packing = set(current_packing)
                candidates = [i for i in range(total_blocks) if i not in temp_packing and i not in remove_indices]
                random.shuffle(candidates)
                
                added = 0
                for idx in candidates:
                    if added == 3:
                        break
                    if not (temp_packing & conflicts[idx]):
                        temp_packing.add(idx)
                        added += 1
                
                if added == 3:
                    current_packing = temp_packing
                    current_size += 1 # Net gain 1
                    improved = True
                else:
                    for r in remove_indices:
                        current_packing.add(r)
                    current_size += 2

        if current_size > best_size:
            best_size = current_size
            best_packing = set(current_packing)
            if best_size >= 12: # Good enough
                break

    num_blocks = best_size
    
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
    for idx in best_packing:
        block_params, _ = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered triples
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
