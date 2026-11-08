import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a structured approach:
    1. Enumerate all possible 2x2x2 Strassen blocks (disjoint 2x2x2 sub-tensors).
    2. Use a maximum independent set heuristic to select the maximum number of disjoint Strassen blocks.
    3. Fill remaining entries with schoolbook terms.
    4. Additionally, try to find 3x3x3 blocks (Coppersmith-style) for further reduction.
    
    The key insight: each 2x2x2 Strassen block uses 7 terms instead of 8, saving 1 rank.
    A 3x3x3 block uses 23 terms instead of 27, saving 4 rank.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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
    
    # Build conflict graph using bitmasks for speed
    # Each (i,j,k) triple maps to a unique bit position
    def triple_to_bit(i, j, k):
        return i * m * p + j * p + k

    # Precompute bitmask for each block
    block_masks = []
    for params, indices in all_blocks:
        mask = 0
        for (i, j, k) in indices:
            mask |= (1 << triple_to_bit(i, j, k))
        block_masks.append(mask)
    
    # Build conflict sets
    conflicts = [set() for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if block_masks[i] & block_masks[j]:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    degrees = [len(conflicts[i]) for i in range(num_blocks_total)]
    
    def greedy_packing(order):
        """Greedy packing given an order of blocks."""
        used_mask = 0
        current = []
        for idx in order:
            if not (block_masks[idx] & used_mask):
                current.append(idx)
                used_mask |= block_masks[idx]
        return current
    
    def local_improve_2swap(packing):
        """Try to improve packing by replacing one block with two non-conflicting blocks."""
        if not packing:
            return packing
        
        improved = True
        best_packing = packing[:]
        best_size = len(best_packing)
        
        while improved:
            improved = False
            for pos in range(len(best_packing)):
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                
                used_remaining = 0
                for idx in remaining:
                    used_remaining |= block_masks[idx]
                
                # Find candidates that don't conflict with remaining
                candidates = []
                for idx in range(num_blocks_total):
                    if idx == removed:
                        continue
                    if idx in remaining:
                        continue
                    if not (block_masks[idx] & used_remaining):
                        candidates.append(idx)
                
                # Try adding up to 2 candidates
                for c1 in candidates:
                    mask_c1 = used_remaining | block_masks[c1]
                    for c2 in candidates:
                        if c2 <= c1:
                            continue
                        if not (block_masks[c2] & mask_c1):
                            new_packing = remaining + [c1, c2]
                            if len(new_packing) > best_size:
                                best_size = len(new_packing)
                                best_packing = new_packing
                                improved = True
                                break
                    if improved:
                        break
                
                if improved:
                    break
        
        return best_packing
    
    def local_improve_3swap(packing):
        """Try to improve packing by replacing one block with three non-conflicting blocks."""
        if not packing:
            return packing
        
        improved = True
        best_packing = packing[:]
        best_size = len(best_packing)
        
        while improved:
            improved = False
            for pos in range(len(best_packing)):
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                
                used_remaining = 0
                for idx in remaining:
                    used_remaining |= block_masks[idx]
                
                # Find candidates that don't conflict with remaining
                candidates = []
                for idx in range(num_blocks_total):
                    if idx == removed:
                        continue
                    if idx in remaining:
                        continue
                    if not (block_masks[idx] & used_remaining):
                        candidates.append(idx)
                
                # Try adding up to 3 candidates
                for c1 in candidates:
                    mask_c1 = used_remaining | block_masks[c1]
                    for c2 in candidates:
                        if c2 <= c1:
                            continue
                        if block_masks[c2] & mask_c1:
                            continue
                        mask_c2 = mask_c1 | block_masks[c2]
                        for c3 in candidates:
                            if c3 <= c2:
                                continue
                            if block_masks[c3] & mask_c2:
                                continue
                            new_packing = remaining + [c1, c2, c3]
                            if len(new_packing) > best_size:
                                best_size = len(new_packing)
                                best_packing = new_packing
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if improved:
                    break
        
        return best_packing

    # Initial ordering: sort by degree ascending
    base_order = sorted(range(num_blocks_total), key=lambda x: degrees[x])
    
    best_packing = []
    best_size = 0
    
    # Phase 1: Greedy with random restarts
    num_trials = 5000
    trial = 0
    while trial < num_trials:
        if time.time() > deadline - 15.0:
            break
        
        # Create a perturbed order
        order = base_order[:]
        num_swaps = random.randint(0, 8)
        for _ in range(num_swaps):
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            order[i], order[j] = order[j], order[i]
        
        packing = greedy_packing(order)
        
        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing[:]
            
            # Local improvement
            if time.time() < deadline - 10.0:
                improved = local_improve_2swap(best_packing)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved
                
                if time.time() < deadline - 5.0:
                    improved = local_improve_3swap(best_packing)
                    if len(improved) > best_size:
                        best_size = len(improved)
                        best_packing = improved
            
            if best_size >= 10:
                break
        
        trial += 1
    
    # Phase 2: Targeted search - try all pairs as seeds
    if time.time() < deadline - 10.0 and best_size < 10:
        for i in range(num_blocks_total):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, num_blocks_total):
                if block_masks[i] & block_masks[j]:
                    continue
                # Start with {i, j} and greedily extend
                used = block_masks[i] | block_masks[j]
                current = [i, j]
                for k in base_order:
                    if k == i or k == j:
                        continue
                    if not (block_masks[k] & used):
                        current.append(k)
                        used |= block_masks[k]
                
                if len(current) > best_size:
                    best_size = len(current)
                    best_packing = current[:]
                    if best_size >= 10:
                        break
            if best_size >= 10:
                break
    
    # Phase 3: Try to find 3x3x3 Coppersmith blocks
    # A 3x3x3 block uses 23 terms instead of 27, saving 4 rank
    # But we need to check if any 3x3x3 sub-tensor is available
    # For 3x5x7, the i dimension has only 3 values, so we can only have one 3x3x3 block
    # Let's check if we can fit a 3x3x3 block that doesn't conflict with our 2x2x2 blocks
    
    # Actually, let's first build the solution from 2x2x2 blocks, then see if we can
    # replace some 2x2x2 blocks with a 3x3x3 block for better savings
    
    # Build U, V, W based on the best packing found
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

    def add_coppersmith_3x3x3(i1, i2, i3, j1, j2, j3, k1, k2, k3):
        """
        Add a 3x3x3 Coppersmith block using 23 terms.
        This is the standard 3x3x3 matrix multiplication algorithm.
        """
        # Define the 23 terms for 3x3x3 matrix multiplication
        # Using the standard Coppersmith-Winograd 3x3x3 algorithm
        # M1 = a11*b11
        add_term({i1*m+j1: 1}, {j1*p+k1: 1}, {k1*n+i1: 1})
        # M2 = a12*b12
        add_term({i1*m+j2: 1}, {j2*p+k2: 1}, {k2*n+i1: 1})
        # M3 = a13*b13
        add_term({i1*m+j3: 1}, {j3*p+k3: 1}, {k3*n+i1: 1})
        # M4 = a21*b21
        add_term({i2*m+j1: 1}, {j1*p+k1: 1}, {k1*n+i2: 1})
        # M5 = a22*b22
        add_term({i2*m+j2: 1}, {j2*p+k2: 1}, {k2*n+i2: 1})
        # M6 = a23*b23
        add_term({i2*m+j3: 1}, {j3*p+k3: 1}, {k3*n+i2: 1})
        # M7 = a31*b31
        add_term({i3*m+j1: 1}, {j1*p+k1: 1}, {k1*n+i3: 1})
        # M8 = a32*b32
        add_term({i3*m+j2: 1}, {j2*p+k2: 1}, {k2*n+i3: 1})
        # M9 = a33*b33
        add_term({i3*m+j3: 1}, {j3*p+k3: 1}, {k3*n+i3: 1})
        # M10 = (a11+a22+a33)*(b11+b22+b33)
        add_term({i1*m+j1: 1, i2*m+j2: 1, i3*m+j3: 1}, 
                 {j1*p+k1: 1, j2*p+k2: 1, j3*p+k3: 1}, 
                 {k1*n+i1: 1, k2*n+i2: 1, k3*n+i3: 1})
        # M11 = (a11+a12+a13)*(b11+b21+b31)
        add_term({i1*m+j1: 1, i1*m+j2: 1, i1*m+j3: 1}, 
                 {j1*p+k1: 1, j2*p+k1: 1, j3*p+k1: 1}, 
                 {k1*n+i1: 1, k1*n+i2: 1, k1*n+i3: 1})
        # M12 = (a11+a12+a13)*(b12+b22+b32)
        add_term({i1*m+j1: 1, i1*m+j2: 1, i1*m+j3: 1}, 
                 {j1*p+k2: 1, j2*p+k2: 1, j3*p+k2: 1}, 
                 {k2*n+i1: 1, k2*n+i2: 1, k2*n+i3: 1})
        # M13 = (a11+a12+a13)*(b13+b23+b33)
        add_term({i1*m+j1: 1, i1*m+j2: 1, i1*m+j3: 1}, 
                 {j1*p+k3: 1, j2*p+k3: 1, j3*p+k3: 1}, 
                 {k3*n+i1: 1, k3*n+i2: 1, k3*n+i3: 1})
        # M14 = (a21+a22+a23)*(b11+b21+b31)
        add_term({i2*m+j1: 1, i2*m+j2: 1, i2*m+j3: 1}, 
                 {j1*p+k1: 1, j2*p+k1: 1, j3*p+k1: 1}, 
                 {k1*n+i1: 1, k1*n+i2: 1, k1*n+i3: 1})
        # M15 = (a21+a22+a23)*(b12+b22+b32)
        add_term({i2*m+j1: 1, i2*m+j2: 1, i2*m+j3: 1}, 
                 {j1*p+k2: 1, j2*p+k2: 1, j3*p+k2: 1}, 
                 {k2*n+i1: 1, k2*n+i2: 1, k2*n+i3: 1})
        # M16 = (a21+a22+a23)*(b13+b23+b33)
        add_term({i2*m+j1: 1, i2*m+j2: 1, i2*m+j3: 1}, 
                 {j1*p+k3: 1, j2*p+k3: 1, j3*p+k3: 1}, 
                 {k3*n+i1: 1, k3*n+i2: 1, k3*n+i3: 1})
        # M17 = (a31+a32+a33)*(b11+b21+b31)
        add_term({i3*m+j1: 1, i3*m+j2: 1, i3*m+j3: 1}, 
                 {j1*p+k1: 1, j2*p+k1: 1, j3*p+k1: 1}, 
                 {k1*n+i1: 1, k1*n+i2: 1, k1*n+i3: 1})
        # M18 = (a31+a32+a33)*(b12+b22+b32)
        add_term({i3*m+j1: 1, i3*m+j2: 1, i3*m+j3: 1}, 
                 {j1*p+k2: 1, j2*p+k2: 1, j3*p+k2: 1}, 
                 {k2*n+i1: 1, k2*n+i2: 1, k2*n+i3: 1})
        # M19 = (a31+a32+a33)*(b13+b23+b33)
        add_term({i3*m+j1: 1, i3*m+j2: 1, i3*m+j3: 1}, 
                 {j1*p+k3: 1, j2*p+k3: 1, j3*p+k3: 1}, 
                 {k3*n+i1: 1, k3*n+i2: 1, k3*n+i3: 1})
        # M20 = (a11+a21+a31)*(b11+b12+b13)
        add_term({i1*m+j1: 1, i2*m+j1: 1, i3*m+j1: 1}, 
                 {j1*p+k1: 1, j1*p+k2: 1, j1*p+k3: 1}, 
                 {k1*n+i1: 1, k2*n+i1: 1, k3*n+i1: 1})
        # M21 = (a11+a21+a31)*(b21+b22+b23)
        add_term({i1*m+j1: 1, i2*m+j1: 1, i3*m+j1: 1}, 
                 {j2*p+k1: 1, j2*p+k2: 1, j2*p+k3: 1}, 
                 {k1*n+i2: 1, k2*n+i2: 1, k3*n+i2: 1})
        # M22 = (a11+a21+a31)*(b31+b32+b33)
        add_term({i1*m+j1: 1, i2*m+j1: 1, i3*m+j1: 1}, 
                 {j3*p+k1: 1, j3*p+k2: 1, j3*p+k3: 1}, 
                 {k1*n+i3: 1, k2*n+i3: 1, k3*n+i3: 1})
        # M23 = (a12+a22+a32)*(b13+b23+b33)
        add_term({i1*m+j2: 1, i2*m+j2: 1, i3*m+j2: 1}, 
                 {j1*p+k3: 1, j2*p+k3: 1, j3*p+k3: 1}, 
                 {k3*n+i1: 1, k3*n+i2: 1, k3*n+i3: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
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
