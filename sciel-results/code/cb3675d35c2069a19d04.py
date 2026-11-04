import time
from itertools import combinations

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    # A block is defined by pairs of indices (i1, i2), (j1, j2), (k1, k2)
    # The indices in the tensor are all combinations of these pairs.
    all_blocks = []
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    # Sort blocks to improve pruning in backtracking
    # Heuristic: Blocks with "rarer" indices might be better to place first?
    # Or just sort by index count (all are 8).
    # Let's try sorting by the sum of indices to break ties deterministically.
    all_blocks.sort(key=lambda x: sum(sum(idx) for idx in x[1]))

    best_num_blocks = 0
    best_packing = []

    # Backtracking search
    # To speed up, we can use a greedy approach to get a good initial bound, then backtrack.
    
    # 1. Greedy to find a good initial packing
    rng_seed = 42
    # We'll do a few greedy passes with different orderings to find a strong lower bound for K
    import random
    rng = random.Random(rng_seed)
    
    for _ in range(100):
        if time.time() > deadline - 20.0:
            break
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        used_indices = set()
        current_packing = []
        for idx in order:
            block_params, block_indices = all_blocks[idx]
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing
            if best_num_blocks >= 12: # High confidence threshold
                break

    # 2. Backtracking to try to improve
    # We need to be careful with recursion depth and time.
    # Since N=630 is small, but branching factor is high, we need good pruning.
    
    def backtrack(idx, used_indices, current_count, current_packing):
        nonlocal best_num_blocks, best_packing
        
        if time.time() > deadline - 10.0:
            return
            
        if current_count + (len(all_blocks) - idx) <= best_num_blocks:
            return
            
        if idx == len(all_blocks):
            if current_count > best_num_blocks:
                best_num_blocks = current_count
                best_packing = list(current_packing)
            return
            
        # Option 1: Skip current block
        backtrack(idx + 1, used_indices, current_count, current_packing)
        
        # Option 2: Include current block if possible
        block_params, block_indices = all_blocks[idx]
        if not (block_indices & used_indices):
            current_packing.append(block_params)
            backtrack(idx + 1, used_indices | block_indices, current_count + 1, current_packing)
            current_packing.pop()

    # The naive backtracking might be too slow if the bound is weak.
    # Let's add a stronger bound.
    # A simple bound: remaining_blocks = len(all_blocks) - idx.
    # Better bound: Count how many blocks among all_blocks[idx:] are disjoint from used_indices.
    # This is expensive to calculate at every step.
    
    # Given the small size (630 blocks), and that we likely can't go much higher than 8-9,
    # and the greedy already found 7, let's try a limited depth or time-boxed search.
    # Actually, 630 is small enough that a well-pruned search might work.
    # But Python recursion is slow.
    
    # Alternative: Use a Max-SAT or ILP solver? Not available in stdlib.
    # Let's stick to the greedy + local search or limited backtracking.
    
    # If backtracking is too slow, we rely on the greedy result.
    # Let's run a short backtracking from the best greedy solution's complement? No, that's complex.
    
    # Let's just run the backtracking with a time check.
    # To make it faster, we can order the blocks dynamically? No, static is fine for now.
    
    # Let's try to run backtrack from the start, but with the initial bound from greedy.
    # This might still be slow.
    
    # Let's assume the greedy result is good enough for now, or try a few more greedy variants.
    # The previous program did 8000 trials. We can do more if time permits.
    
    # Let's increase the number of greedy trials if time is available.
    trials_done = 100
    while time.time() < deadline - 20.0 and trials_done < 5000:
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        used_indices = set()
        current_packing = []
        for idx in order:
            block_params, block_indices = all_blocks[idx]
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing
            if best_num_blocks >= 12:
                break
        trials_done += 1

    # Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        for idx, val in u_entries.items():
            u[idx] = val
        for idx, val in v_entries.items():
            v[idx] = val
        for idx, val in w_entries.items():
            w[idx] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

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

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 5.0:
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
