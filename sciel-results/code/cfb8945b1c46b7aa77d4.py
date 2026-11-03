import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 5, 7
    
    # All possible 2x2x2 blocks: (i0,i1,j0,j1,k0,k1)
    # i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    i_pairs = [(0,1), (0,2), (1,2)]
    j_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    k_pairs = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
               (1,2), (1,3), (1,4), (1,5), (1,6),
               (2,3), (2,4), (2,5), (2,6),
               (3,4), (3,5), (3,6),
               (4,5), (4,6),
               (5,6)]
    
    # Generate all possible blocks
    all_blocks = []
    for (i0, i1) in i_pairs:
        for (j0, j1) in j_pairs:
            for (k0, k1) in k_pairs:
                # This block covers (i,j,k) for i in {i0,i1}, j in {j0,j1}, k in {k0,k1}
                covered = set()
                for i in (i0, i1):
                    for j in (j0, j1):
                        for k in (k0, k1):
                            covered.add((i, j, k))
                all_blocks.append(((i0, i1, j0, j1, k0, k1), covered))
    
    # Find maximum set of disjoint blocks
    # Use greedy: sort by coverage size (all same), try to maximize count
    # Actually all blocks cover exactly 8 cells, so we want max number of disjoint blocks
    
    # Greedy approach: try all single blocks, then add non-overlapping ones
    # For small search space, use a simple greedy with multiple restarts
    
    def find_max_disjoint(blocks):
        """Find max set of pairwise disjoint blocks using greedy + local search."""
        best_count = 0
        best_set = []
        
        # Sort blocks to try different orderings
        # Try greedy with different starting blocks
        for start_idx in range(min(50, len(blocks))):  # limit for speed
            chosen = [start_idx]
            used_cells = set(blocks[start_idx][1])
            for other_idx in range(len(blocks)):
                if other_idx == start_idx:
                    continue
                if not used_cells & blocks[other_idx][1]:
                    chosen.append(other_idx)
                    used_cells |= blocks[other_idx][1]
            if len(chosen) > best_count:
                best_count = len(chosen)
                best_set = chosen[:]
        
        # Local search: try swapping
        improved = True
        while improved:
            improved = False
            current_set = set(best_set)
            for idx in best_set:
                # Try removing idx and adding a different block
                for other_idx in range(len(blocks)):
                    if other_idx in current_set:
                        continue
                    # Check if other_idx is disjoint from all blocks in best_set except idx
                    rest = [b for b in best_set if b != idx]
                    compatible = True
                    for b in rest:
                        if blocks[other_idx][1] & blocks[b][1]:
                            compatible = False
                            break
                    if compatible:
                        new_set = rest + [other_idx]
                        # This doesn't increase count, but might enable further additions
                        # Skip for now
                        pass
            
            # Try adding any block not in set
            for other_idx in range(len(blocks)):
                if other_idx in current_set:
                    continue
                compatible = True
                for b in best_set:
                    if blocks[other_idx][1] & blocks[b][1]:
                        compatible = False
                        break
                if compatible:
                    best_set.append(other_idx)
                    best_count += 1
                    improved = True
                    break
        
        return best_set, best_count
    
    # Check time
    if time.time() > deadline - 30:
        # Just use 1 block
        best_set = [0]
        best_count = 1
    else:
        best_set, best_count = find_max_disjoint(all_blocks)
    
    # Build the decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    covered = set()
    
    for block_idx in best_set:
        (i0, i1, j0, j1, k0, k1) = all_blocks[block_idx][0]
        covered |= all_blocks[block_idx][1]
        
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1
        
        # Strassen 7 terms
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
    
    # Schoolbook for all other entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
    
    rank = len(U_cols)
    
    # Build matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
    
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
        # Fallback to pure schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
