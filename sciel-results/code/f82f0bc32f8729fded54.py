"""Deterministic block-decomposition search for 2x5x6 matrix multiplication tensor."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Helper to validate a decomposition (vectorized)
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Build tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # T should have T[i,j,k] = 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    # Schoolbook baseline: rank = n*m*p = 60
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_sb

    # STRATTEN-BASED BLOCK DECOMPOSITION
    # Decompose 2x5x6 into 2x1x6 (rank 12) and 2x4x6.
    # 2x4x6 can be decomposed into two 2x2x6 blocks.
    # Each 2x2x6 block can be handled by 3 Strassen instances (2x2x2) each of rank 7.
    # Total rank = 12 + 7*3*2 = 54.
    
    # Let's construct this explicitly to guarantee a valid rank 54 solution.
    # This serves as a strong baseline to beat with the greedy search if possible,
    # or as the fallback.
    
    U_strassen = np.zeros((n * m, 54))
    V_strassen = np.zeros((m * p, 54))
    W_strassen = np.zeros((p * n, 54))
    
    col = 0
    
    # Block 1: 2x1x6 (A[:, 0] * B[0, :]) -> Rank 12
    # A is 2x1, B is 1x6. C is 2x6.
    # C[i, k] = A[i, 0] * B[0, k]
    for i in range(n):
        for k in range(p):
            U_strassen[i * m + 0, col] = 1
            V_strassen[0 * p + k, col] = 1
            W_strassen[k * n + i, col] = 1
            col += 1
    # col is now 12

    # Block 2: 2x2x6 (A[:, 1:3] * B[1:3, :]) -> Rank 21 (3 Strassen blocks of size 2x2x2)
    # A indices: i in {0,1}, j in {1,2}
    # B indices: j in {1,2}, k in {0..5}
    # Split k into {0,1}, {2,3}, {4,5}
    
    def add_strassen_block(i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting indices for the 2x2x2 block
        # i: i_base, i_base+1
        # j: j_base, j_base+1
        # k: k_base, k_base+1
        
        # We need to map local indices (0,1) to global indices
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        # M1 = (A11-A22)*(B11+B22)
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # A11 is (i_base, j_base), A22 is (i_base+1, j_base+1)
        u[i_base * m + j_base] = 1
        u[(i_base+1) * m + (j_base+1)] = -1
        
        # B11 is (j_base, k_base), B22 is (j_base+1, k_base+1)
        v[j_base * p + k_base] = 1
        v[(j_base+1) * p + (k_base+1)] = 1
        
        # W indices for C11, C12, C21, C22
        # C11: (i_base, k_base) -> idx k_base*n + i_base
        # C12: (i_base, k_base+1) -> idx (k_base+1)*n + i_base
        # C21: (i_base+1, k_base) -> idx k_base*n + i_base+1
        # C22: (i_base+1, k_base+1) -> idx (k_base+1)*n + i_base+1
        
        w[k_base * n + i_base] = 1
        w[(k_base+1) * n + i_base] = 1
        w[k_base * n + (i_base+1)] = 1
        w[(k_base+1) * n + (i_base+1)] = 1
        
        return u, v, w
        
    # M2 = (A21+A22)*B11
    def get_m2(i_base, j_base, k_base):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # A21 is (i_base+1, j_base), A22 is (i_base+1, j_base+1)
        u[(i_base+1) * m + j_base] = 1
        u[(i_base+1) * m + (j_base+1)] = 1
        
        # B11 is (j_base, k_base)
        v[j_base * p + k_base] = 1
        
        # C21, C22
        w[k_base * n + (i_base+1)] = 1
        w[(k_base+1) * n + (i_base+1)] = 1
        
        return u, v, w

    # M3 = A11*(B21-B22)
    def get_m3(i_base, j_base, k_base):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # A11 is (i_base, j_base)
        u[i_base * m + j_base] = 1
        
        # B21 is (j_base+1, k_base), B22 is (j_base+1, k_base+1)
        v[(j_base+1) * p + k_base] = 1
        v[(j_base+1) * p + (k_base+1)] = -1
        
        # C11, C12
        w[k_base * n + i_base] = 1
        w[(k_base+1) * n + i_base] = 1
        
        return u, v, w

    # M4 = A22*(B12-B11)
    def get_m4(i_base, j_base, k_base):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # A22 is (i_base+1, j_base+1)
        u[(i_base+1) * m + (j_base+1)] = 1
        
        # B12 is (j_base, k_base+1), B11 is (j_base, k_base)
        v[j_base * p + (k_base+1)] = 1
        v[j_base * p + k_base] = -1
        
        # C21, C22
        w[k_base * n + (i_base+1)] = 1
        w[(k_base+1) * n + (i_base+1)] = 1
        
        return u, v, w

    # M5 = (A11+A12)*B22
    def get_m5(i_base, j_base, k_base):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # A11 is (i_base, j_base), A12 is (i_base, j_base+1)
        u[i_base * m + j_base] = 1
        u[i_base * m + (j_base+1)] = 1
        
        # B22 is (j_base+1, k_base+1)
        v[(j_base+1) * p + (k_base+1)] = 1
        
        # C12
        w[(k_base+1) * n + i_base] = 1
        
        return u, v, w

    # M6 = (A21-A11)*(B11+B12)
    def get_m6(i_base, j_base, k_base):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # A21 is (i_base+1, j_base), A11 is (i_base, j_base)
        u[(i_base+1) * m + j_base] = 1
        u[i_base * m + j_base] = -1
        
        # B11 is (j_base, k_base), B12 is (j_base, k_base+1)
        v[j_base * p + k_base] = 1
        v[j_base * p + (k_base+1)] = 1
        
        # C21, C22
        w[k_base * n + (i_base+1)] = 1
        w[(k_base+1) * n + (i_base+1)] = 1
        
        return u, v, w

    # M7 = (A12-A22)*(B21+B22)
    def get_m7(i_base, j_base, k_base):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # A12 is (i_base, j_base+1), A22 is (i_base+1, j_base+1)
        u[i_base * m + (j_base+1)] = 1
        u[(i_base+1) * m + (j_base+1)] = -1
        
        # B21 is (j_base+1, k_base), B22 is (j_base+1, k_base+1)
        v[(j_base+1) * p + k_base] = 1
        v[(j_base+1) * p + (k_base+1)] = 1
        
        # C11, C12
        w[k_base * n + i_base] = 1
        w[(k_base+1) * n + i_base] = 1
        
        return u, v, w

    # Add Strassen blocks for 2x2x6 part (j=1,2)
    # k splits: 0-1, 2-3, 4-5
    for k_start in [0, 2, 4]:
        u1, v1, w1 = add_strassen_block(0, 1, k_start)
        U_strassen[:, col] = u1; V_strassen[:, col] = v1; W_strassen[:, col] = w1; col += 1
        u2, v2, w2 = get_m2(0, 1, k_start)
        U_strassen[:, col] = u2; V_strassen[:, col] = v2; W_strassen[:, col] = w2; col += 1
        u3, v3, w3 = get_m3(0, 1, k_start)
        U_strassen[:, col] = u3; V_strassen[:, col] = v3; W_strassen[:, col] = w3; col += 1
        u4, v4, w4 = get_m4(0, 1, k_start)
        U_strassen[:, col] = u4; V_strassen[:, col] = v4; W_strassen[:, col] = w4; col += 1
        u5, v5, w5 = get_m5(0, 1, k_start)
        U_strassen[:, col] = u5; V_strassen[:, col] = v5; W_strassen[:, col] = w5; col += 1
        u6, v6, w6 = get_m6(0, 1, k_start)
        U_strassen[:, col] = u6; V_strassen[:, col] = v6; W_strassen[:, col] = w6; col += 1
        u7, v7, w7 = get_m7(0, 1, k_start)
        U_strassen[:, col] = u7; V_strassen[:, col] = v7; W_strassen[:, col] = w7; col += 1
        
    # col should be 12 + 21 = 33
    
    # Block 3: 2x2x6 (A[:, 3:5] * B[3:5, :]) -> Rank 21
    for k_start in [0, 2, 4]:
        u1, v1, w1 = add_strassen_block(0, 3, k_start)
        U_strassen[:, col] = u1; V_strassen[:, col] = v1; W_strassen[:, col] = w1; col += 1
        u2, v2, w2 = get_m2(0, 3, k_start)
        U_strassen[:, col] = u2; V_strassen[:, col] = v2; W_strassen[:, col] = w2; col += 1
        u3, v3, w3 = get_m3(0, 3, k_start)
        U_strassen[:, col] = u3; V_strassen[:, col] = v3; W_strassen[:, col] = w3; col += 1
        u4, v4, w4 = get_m4(0, 3, k_start)
        U_strassen[:, col] = u4; V_strassen[:, col] = v4; W_strassen[:, col] = w4; col += 1
        u5, v5, w5 = get_m5(0, 3, k_start)
        U_strassen[:, col] = u5; V_strassen[:, col] = v5; W_strassen[:, col] = w5; col += 1
        u6, v6, w6 = get_m6(0, 3, k_start)
        U_strassen[:, col] = u6; V_strassen[:, col] = v6; W_strassen[:, col] = w6; col += 1
        u7, v7, w7 = get_m7(0, 3, k_start)
        U_strassen[:, col] = u7; V_strassen[:, col] = v7; W_strassen[:, col] = w7; col += 1
        
    # col should be 33 + 21 = 54

    if is_valid_decomposition(U_strassen, V_strassen, W_strassen, col):
        best_U = U_strassen[:n*m, :col].copy()
        best_V = V_strassen[:m*p, :col].copy()
        best_W = W_strassen[:p*n, :col].copy()
        best_rank = col

    # Now, try to improve upon this using a greedy search with a larger candidate pool.
    # We will use the rank-54 solution as a starting point for the greedy search?
    # No, the greedy search builds from scratch. But we can use the candidate pool
    # generated from the Strassen blocks and standard blocks.
    
    # Candidate Pool Generation
    candidates = []
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
                
    # Type 2: Strassen-inspired candidates for all 2x2x2 blocks
    # We already have the functions above. Let's add all possible 2x2x2 blocks.
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                u1, v1, w1 = add_strassen_block(i0, j0, k0)
                candidates.append((u1, v1, w1))
                u2, v2, w2 = get_m2(i0, j0, k0)
                candidates.append((u2, v2, w2))
                u3, v3, w3 = get_m3(i0, j0, k0)
                candidates.append((u3, v3, w3))
                u4, v4, w4 = get_m4(i0, j0, k0)
                candidates.append((u4, v4, w4))
                u5, v5, w5 = get_m5(i0, j0, k0)
                candidates.append((u5, v5, w5))
                u6, v6, w6 = get_m6(i0, j0, k0)
                candidates.append((u6, v6, w6))
                u7, v7, w7 = get_m7(i0, j0, k0)
                candidates.append((u7, v7, w7))

    # Build the linear system matrix
    num_constraints = n * m * p
    A_mat = np.zeros((num_constraints, len(candidates)))
    b_vec = np.ones(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c_idx, (u, v, w) in enumerate(candidates):
                    A_mat[idx, c_idx] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                idx += 1
    
    # Greedy approach: iteratively select candidates that best reduce residual
    # Start from the best known rank (54) and see if we can go lower?
    # The greedy search might find a solution with rank < 54 if one exists in the pool.
    # However, the pool is limited.
    # Let's run the greedy search to see if it finds anything better than 54.
    
    residual = b_vec.copy()
    selected = []
    
    # To speed up, we can pre-compute dot products or use vectorized operations.
    # But len(candidates) is small (~60 standard + 4*4*5*7 strassen = 60 + 560 = 620).
    # 620 candidates, 60 constraints.
    
    for step in range(min(best_rank, len(candidates))):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Find candidate that best reduces residual
        best_c = -1
        best_reduction = -np.inf
        
        # Vectorized calculation of optimal x_c for each candidate
        # x_c = <r, a_c> / <a_c, a_c>
        # reduction = ||r||^2 - ||r - x_c a_c||^2 = x_c^2 * <a_c, a_c>
        
        dots_ra = residual @ A_mat  # shape (num_candidates,)
        dots_aa = np.sum(A_mat * A_mat, axis=0)  # shape (num_candidates,)
        
        # Avoid division by zero
        valid = dots_aa > 1e-10
        x_vals = np.zeros_like(dots_ra)
        x_vals[valid] = dots_ra[valid] / dots_aa[valid]
        
        reductions = x_vals * dots_aa
        
        # Mask out already selected
        if selected:
            reductions[selected] = -np.inf
            
        best_c = np.argmax(reductions)
        best_reduction = reductions[best_c]
        
        if best_c == -1 or best_reduction < 1e-10:
            break
        
        selected.append(best_c)
        residual = residual - x_vals[best_c] * A_mat[:, best_c]
        
        # Check if we've converged
        if np.max(np.abs(residual)) < 1e-8:
            break
    
    # Verify the selected candidates form a valid decomposition
    if len(selected) > 0 and len(selected) < best_rank:
        U_sel = np.zeros((n * m, len(selected)))
        V_sel = np.zeros((m * p, len(selected)))
        W_sel = np.zeros((p * n, len(selected)))
        
        for sel_idx, c_idx in enumerate(selected):
            u, v, w = candidates[c_idx]
            U_sel[:, sel_idx] = u
            V_sel[:, sel_idx] = v
            W_sel[:, sel_idx] = w
        
        if is_valid_decomposition(U_sel, V_sel, W_sel, len(selected)):
            best_U = U_sel.copy()
            best_V = V_sel.copy()
            best_W = W_sel.copy()
            best_rank = len(selected)

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
