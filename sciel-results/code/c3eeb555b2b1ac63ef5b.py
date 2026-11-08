import time
import random
import numpy as np

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

    num_blocks = len(all_blocks)
    
    # Precompute bitmask representation for each block's indices
    # Map each (i,j,k) to a bit position
    triple_to_bit = {}
    bit_to_triple = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triple_to_bit[(i, j, k)] = len(bit_to_triple)
                bit_to_triple.append((i, j, k))
    
    num_triples = n * m * p  # 105
    # Use Python integers as bitmasks (arbitrary precision)
    block_bitmasks = []
    for _, indices in all_blocks:
        mask = 0
        for idx in indices:
            mask |= (1 << triple_to_bit[idx])
        block_bitmasks.append(mask)
    
    # Build conflict graph using bitmasks
    conflicts = [0] * num_blocks  # bitmask of conflicting blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if block_bitmasks[i] & block_bitmasks[j]:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    degrees = [bin(conflicts[i]).count('1') for i in range(num_blocks)]
    
    # Greedy packing using bitmask for used indices
    def greedy_packing(order):
        used_mask = 0
        current = []
        for idx in order:
            if not (block_bitmasks[idx] & used_mask):
                current.append(idx)
                used_mask |= block_bitmasks[idx]
        return current

    # Local improvement: try to replace one block with two non-conflicting blocks
    def local_improve(packing):
        if not packing:
            return packing
        
        best_packing = packing[:]
        improved = True
        while improved:
            improved = False
            for pos in range(len(best_packing)):
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                
                # Compute used mask for remaining blocks
                used_mask = 0
                for b in remaining:
                    used_mask |= block_bitmasks[b]
                
                # Find all blocks that don't conflict with remaining and fit
                available = []
                for b in range(num_blocks):
                    if b in remaining or b == removed:
                        continue
                    # Check if b conflicts with any block in remaining
                    # Using conflict bitmask: if any bit in conflicts[b] is set in the remaining mask
                    remaining_mask = 0
                    for r in remaining:
                        remaining_mask |= (1 << r)
                    if conflicts[b] & remaining_mask:
                        continue
                    # Check if b fits (no index overlap with used)
                    if block_bitmasks[b] & used_mask:
                        continue
                    available.append(b)
                
                if not available:
                    continue
                
                # Try adding pairs of available blocks
                best_pair = None
                best_size = len(best_packing)
                for i in range(len(available)):
                    for j in range(i + 1, len(available)):
                        b1, b2 = available[i], available[j]
                        # Check if b1 and b2 are compatible (no conflict, no index overlap)
                        if conflicts[b1] & (1 << b2):
                            continue
                        if block_bitmasks[b1] & block_bitmasks[b2]:
                            continue
                        # Both fit
                        if (block_bitmasks[b1] | block_bitmasks[b2]) & used_mask:
                            continue
                        # This pair works
                        if best_pair is None or len(remaining) + 2 > best_size:
                            best_pair = (b1, b2)
                            best_size = len(remaining) + 2
                
                if best_pair is not None:
                    new_packing = remaining + list(best_pair)
                    if len(new_packing) > len(best_packing):
                        best_packing = new_packing
                        improved = True
                        break
        
        return best_packing

    best_packing = []
    best_size = 0

    # Multiple restarts with optimized search
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        # Heuristic ordering: sort by degree (ascending) to pick less constrained blocks first
        order = list(range(num_blocks))
        random.shuffle(order)
        order.sort(key=lambda x: (degrees[x], random.random()))
        
        packing = greedy_packing(order)
        
        # Local improvement
        packing = local_improve(packing)
        
        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing[:]
            
            # If we found a good packing, try more aggressive local search
            if best_size >= 12:
                # Do additional rounds of local improvement
                for _ in range(3):
                    new_packing = local_improve(best_packing)
                    if len(new_packing) > len(best_packing):
                        best_packing = new_packing
                        best_size = len(best_packing)
                    else:
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
    for idx in best_packing:
        params, _ = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = params
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
