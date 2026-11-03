import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 3, 5, 7
    N, M, P = n * m, m * p, p * n  # 15, 35, 21

    # Bini rank-19 decomposition for 3x3x3
    # Indices: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # U: (i,j) -> list of (term_index, coeff)
    # V: (j,k) -> list of (term_index, coeff)
    # W: (k,i) -> list of (term_index, coeff)
    
    # Standard Bini 3x3 rank-19
    # Terms 0..18
    bini_U = [
        [(0,1),(1,1),(2,1)],  # (0,0)
        [(1,1),(3,1)],         # (0,1)
        [(2,1),(4,1)],         # (0,2)
        [(1,1),(5,1)],         # (1,0)
        [(3,1),(6,1)],         # (1,1)
        [(4,1),(7,1)],         # (1,2)
        [(2,1),(8,1)],         # (2,0)
        [(7,1),(9,1)],         # (2,1)
        [(8,1),(10,1)],        # (2,2)
    ]
    bini_V = [
        [(0,1),(1,1),(2,1)],  # (0,0)
        [(1,1),(3,1)],         # (0,1)
        [(2,1),(4,1)],         # (0,2)
        [(1,1),(5,1)],         # (1,0)
        [(3,1),(6,1)],         # (1,1)
        [(4,1),(7,1)],         # (1,2)
        [(2,1),(8,1)],         # (2,0)
        [(7,1),(9,1)],         # (2,1)
        [(8,1),(10,1)],        # (2,2)
    ]
    bini_W = [
        [(0,1),(1,1),(2,1)],  # (0,0)
        [(1,1),(3,1)],         # (0,1)
        [(2,1),(4,1)],         # (0,2)
        [(1,1),(5,1)],         # (1,0)
        [(3,1),(6,1)],         # (1,1)
        [(4,1),(7,1)],         # (1,2)
        [(2,1),(8,1)],         # (2,0)
        [(7,1),(9,1)],         # (2,1)
        [(8,1),(10,1)],        # (2,2)
    ]
    
    # Actually, the Bini 3x3 rank-19 is more complex. Let me use a known correct one.
    # The standard Bini 3x3 rank-19 decomposition:
    # U (15x19), V (15x19), W (15x19) for 3x3 matmul
    # Let me just use the known rank-19 for 3x3.
    
    # Actually, let me reconsider. The Bini 3x3 rank-19 is non-trivial to hardcode correctly.
    # Let me use a simpler approach: use Strassen 2x2x2 blocks and schoolbook for the rest.
    # This is what the current program does. Let me optimize the block packing.
    
    # For 3x5x7:
    # i in {0,1,2}, j in {0..4}, k in {0..6}
    # Total cells: 3*5*7 = 105
    
    # Strassen 2x2x2: 7 terms per block, covers 8 cells
    # Schoolbook: 1 term per cell
    
    # Strategy: pack as many disjoint 2x2x2 Strassen blocks as possible,
    # then use schoolbook for remaining cells.
    
    # A 2x2x2 block is defined by (i0,i1,j0,j1,k0,k1) where i1=i0+1, j1=j0+1, k1=k0+1
    # Wait, no. The block can be any 2x2x2 subcube, not necessarily adjacent.
    # But for Strassen, we need a 2x2x2 sub-tensor. The indices don't need to be adjacent
    # in the original matrix, but the decomposition works on any 2x2x2 sub-tensor.
    
    # Actually, for the tensor decomposition, we can decompose any 2x2x2 sub-tensor
    # (i.e., any choice of 2 i-values, 2 j-values, 2 k-values) using Strassen's 7 terms.
    
    # So we want to pack as many disjoint 2x2x2 sub-tensors as possible.
    # Each block uses 2 i-values, 2 j-values, 2 k-values.
    # Two blocks are disjoint if they don't share any (i,j,k) triple.
    
    # Let's enumerate all possible 2x2x2 blocks and find max disjoint set.
    
    i_vals = [0, 1, 2]
    j_vals = [0, 1, 2, 3, 4]
    k_vals = [0, 1, 2, 3, 4, 5, 6]
    
    from itertools import combinations
    
    i_pairs = list(combinations(i_vals, 2))
    j_pairs = list(combinations(j_vals, 2))
    k_pairs = list(combinations(k_vals, 2))
    
    all_blocks = []
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                covered = set()
                for i in (i0, i1):
                    for j in (j0, j1):
                        for k in (k0, k1):
                            covered.add((i, j, k))
                all_blocks.append(((i0, i1, j0, j1, k0, k1), covered))
    
    # Find max disjoint set using greedy with multiple restarts
    def find_max_disjoint(blocks, max_starts=100):
        best_count = 0
        best_set = []
        
        for start_idx in range(min(max_starts, len(blocks))):
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
        
        return best_set, best_count
    
    if time.time() > deadline - 60:
        best_set = [0]
        best_count = 1
    else:
        best_set, best_count = find_max_disjoint(all_blocks, max_starts=200)
    
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
