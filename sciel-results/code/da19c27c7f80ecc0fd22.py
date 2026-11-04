import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    
    # Generate all possible 2x2x2 blocks
    # A block is defined by (i1, i2, j1, j2, k1, k2) with i1 < i2, j1 < j2, k1 < k2
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # The set of (i,j,k) indices covered by this block
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    # Greedy search for maximum disjoint packing with multiple strategies
    best_num_blocks = 0
    best_packing = []
    
    # Pre-compute block sizes for weighted greedy
    block_sizes = [len(block_indices) for _, block_indices in all_blocks]
    
    rng = random.Random(42)
    
    # Strategy 1: Randomized greedy with different shuffling
    num_trials = 3000
    for t in range(num_trials):
        if time.time() > deadline - 15.0:
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
            if best_num_blocks >= 14:
                break
    
    # Strategy 2: Weighted greedy - prefer blocks that cover "scarce" dimensions
    # i-dimension has only 3 values, so blocks sharing i indices are more constrained
    # Try ordering by i-pair rarity
    for t in range(500):
        if time.time() > deadline - 15.0:
            break
        
        # Score blocks: prefer those with less common i-pairs
        # i-pairs: (0,1), (0,2), (1,2) - all equally common, so use j,k pair diversity
        scores = []
        for idx, (params, indices) in enumerate(all_blocks):
            i1, i2, j1, j2, k1, k2 = params
            # Prefer blocks with more spread in j and k
            score = (j2 - j1) + (k2 - k1)
            scores.append((score, idx))
        
        # Sort by score descending, with random tie-breaking
        rng.shuffle(scores)
        scores.sort(key=lambda x: -x[0])
        
        used_indices = set()
        current_packing = []
        
        for score, idx in scores:
            block_params, block_indices = all_blocks[idx]
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
                
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing
            if best_num_blocks >= 14:
                break
    
    # Strategy 3: Local search - start from best, try to add blocks by removing conflicting ones
    if best_num_blocks > 0 and time.time() < deadline - 20.0:
        for _ in range(200):
            if time.time() > deadline - 15.0:
                break
            
            # Try to improve current packing
            used_indices = set()
            current_packing = list(best_packing)
            for params in current_packing:
                for i in (params[0], params[1]):
                    for j in (params[2], params[3]):
                        for k in (params[4], params[5]):
                            used_indices.add((i, j, k))
            
            # Try adding any block that fits
            improved = True
            while improved:
                improved = False
                for idx in range(len(all_blocks)):
                    if time.time() > deadline - 15.0:
                        break
                    block_params, block_indices = all_blocks[idx]
                    if not (block_indices & used_indices):
                        # Check if already in packing
                        if block_params not in current_packing:
                            current_packing.append(block_params)
                            used_indices |= block_indices
                            improved = True
                            if len(current_packing) > best_num_blocks:
                                best_num_blocks = len(current_packing)
                                best_packing = list(current_packing)

    # Construct U, V, W based on the best packing found
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
        # Map local indices to global row indices
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
    
    # Debug: print packing info
    # print(f"Best packing: {best_num_blocks} blocks, rank so far: {len(U_cols)}")

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format (list of lists)
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
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
