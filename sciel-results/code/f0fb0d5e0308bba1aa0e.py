"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Helper to add a Strassen block at offsets (i0, j0, k0)
    def add_strassen_block(U, V, W, i0, j0, k0, start_col):
        c = start_col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = start_col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = start_col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        
        return start_col + 7
    
    def build_decomposition(blocks):
        num_blocks = len(blocks)
        strassen_rank = num_blocks * 7
        schoolbook_rank = n * m * p - num_blocks * 8
        total_rank = strassen_rank + schoolbook_rank
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, j0, k0) in blocks:
            col = add_strassen_block(U, V, W, i0, j0, k0, col)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    covered = False
                    for (i0, j0, k0) in blocks:
                        if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                            covered = True
                            break
                    if not covered:
                        U[i*m + j][col] += 1
                        V[j*p + k][col] += 1
                        W[k*n + i][col] += 1
                        col += 1
        
        return {"U": U, "V": V, "W": W}, total_rank
    
    def verify_decomp(result):
        U_arr = np.array(result["U"])
        V_arr = np.array(result["V"])
        W_arr = np.array(result["W"])
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        expected = np.ones((n, m, p))
        
        return np.allclose(T, expected, atol=1e-10)
    
    # Try the standard 4-block configuration
    standard_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    result, rank = build_decomposition(standard_blocks)
    
    if not verify_decomp(result):
        # Fallback to schoolbook
        U_fb = [[0.0] * (n*m*p) for _ in range(n*m)]
        V_fb = [[0.0] * (n*m*p) for _ in range(m*p)]
        W_fb = [[0.0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1.0
                    V_fb[j*p+k][col] = 1.0
                    W_fb[k*n+i][col] = 1.0
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_result = result
    best_rank = rank
    
    # Try to find better configurations by searching for overlapping or different block placements
    # Since we can only use i=0,1 for Strassen, and need disjoint j,k pairs,
    # let's try all possible combinations of 2 j-pairs and 2 k-pairs
    
    j_pairs = [(0,1), (1,2), (2,3), (3,4)]
    k_pairs = [(0,1), (1,2), (2,3), (3,4)]
    
    # We need 2 disjoint j-pairs and 2 disjoint k-pairs
    def get_disjoint_pairs(pairs, count=2):
        from itertools import combinations
        results = []
        for combo in combinations(range(len(pairs)), count):
            selected = [pairs[i] for i in combo]
            # Check if disjoint
            all_indices = set()
            disjoint = True
            for pair in selected:
                if all_indices & set(pair):
                    disjoint = False
                    break
                all_indices.update(pair)
            if disjoint:
                results.append(selected)
        return results
    
    j_options = get_disjoint_pairs(j_pairs)
    k_options = get_disjoint_pairs(k_pairs)
    
    # Try all combinations
    for j_sel in j_options:
        for k_sel in k_options:
            blocks = []
            for j0, j1 in j_sel:
                for k0, k1 in k_sel:
                    blocks.append((0, j0, k0))
            
            if len(blocks) != 4:
                continue
                
            res, r = build_decomposition(blocks)
            if verify_decomp(res) and r < best_rank:
                best_result = res
                best_rank = r
    
    # Try with i=1 instead of i=0 (should be symmetric)
    j_pairs2 = [(0,1), (1,2), (2,3), (3,4)]
    k_pairs2 = [(0,1), (1,2), (2,3), (3,4)]
    
    j_options2 = get_disjoint_pairs(j_pairs2)
    k_options2 = get_disjoint_pairs(k_pairs2)
    
    for j_sel in j_options2:
        for k_sel in k_options2:
            blocks = []
            for j0, j1 in j_sel:
                for k0, k1 in k_sel:
                    blocks.append((1, j0, k0))
            
            if len(blocks) != 4:
                continue
                
            res, r = build_decomposition(blocks)
            if verify_decomp(res) and r < best_rank:
                best_result = res
                best_rank = r
    
    return best_result
# EVOLVE-BLOCK-END
