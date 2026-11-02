"""
6x6 matrix multiplication tensor decomposition.
Strategy: Use a 3x3 block partition of 2x2 blocks, applying Strassen's 7-product
decomposition to each pair of 2x2 blocks. This gives rank 3^3 * 7 = 189.
Then attempt to eliminate redundant columns via tensor-level linear algebra.
"""

import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (36, rank), (36, rank), (36, rank).
    """
    t0 = time.time()
    deadline = t0 + 120  # 120s safety margin
    
    n, m, p = 6, 6, 6
    
    # Strassen's 7-product decomposition for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each of the 7 Strassen products, define:
    # u_coeffs: list of (block_row, block_col, coeff) for A
    # v_coeffs: list of (block_row, block_col, coeff) for B  
    # w_coeffs: list of (block_row, block_col, coeff) for C
    # where block indices are in {0,1} for the 2x2 sub-blocks
    
    strassen = [
        # M0: (A00+A11)(B00+B11)
        {
            'u': [(0,0,1), (1,1,1)],
            'v': [(0,0,1), (1,1,1)],
            'w': [(0,0,1), (1,1,1)]
        },
        # M1: (A10+A11)B00
        {
            'u': [(1,0,1), (1,1,1)],
            'v': [(0,0,1)],
            'w': [(1,0,1), (1,1,-1)]
        },
        # M2: A00(B01-B11)
        {
            'u': [(0,0,1)],
            'v': [(0,1,1), (1,1,-1)],
            'w': [(0,1,1), (1,1,1)]
        },
        # M3: A11(B10-B00)
        {
            'u': [(1,1,1)],
            'v': [(1,0,1), (0,0,-1)],
            'w': [(0,0,1), (1,0,1)]
        },
        # M4: (A00+A01)B11
        {
            'u': [(0,0,1), (0,1,1)],
            'v': [(1,1,1)],
            'w': [(0,0,-1), (0,1,1)]
        },
        # M5: (A10-A00)(B00+B10)
        {
            'u': [(1,0,1), (0,0,-1)],
            'v': [(0,0,1), (1,0,1)],
            'w': [(1,1,1)]
        },
        # M6: (A01-A11)(B01+B11)
        {
            'u': [(0,1,1), (1,1,-1)],
            'v': [(0,1,1), (1,1,1)],
            'w': [(0,0,1)]
        },
    ]
    
    nb = 3  # 3x3 grid of blocks
    bs = 2  # block size
    
    rank = nb * nb * nb * 7  # 189
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for ib in range(nb):  # output block row
        for kb in range(nb):  # output block col
            for jb in range(nb):  # input block col (shared between A and B)
                for s in range(7):
                    # U: A block at (ib, jb)
                    for (lr, lc, coeff) in strassen[s]['u']:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U[gi * m + gj, col] += coeff
                    
                    # V: B block at (jb, kb)
                    for (lr, lc, coeff) in strassen[s]['v']:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V[gj * p + gk, col] += coeff
                    
                    # W: C block at (ib, kb)
                    for (lr, lc, coeff) in strassen[s]['w']:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W[gk * n + gi, col] += coeff
                    
                    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        rng = np.random.default_rng(12345)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(U_mat.shape[1]):
                    w_val = W_mat[k*n+i, r]
                    if abs(w_val) < 1e-15:
                        continue
                    u_val = 0.0
                    for j in range(m):
                        u_val += U_mat[i*m+j, r] * A[i,j]
                    v_val = 0.0
                    for j in range(m):
                        v_val += V_mat[j*p+k, r] * B[j,k]
                    C_computed[i, k] += w_val * u_val * v_val
        
        return np.allclose(C_computed, C_expected, atol=tol)
    
    if not verify(U, V, W):
        # Fallback to schoolbook (rank 216)
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.float64)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.float64)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, c] = 1.0
                    V_sb[j*p+k, c] = 1.0
                    W_sb[k*n+i, c] = 1.0
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    if time.time() > deadline - 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt rank reduction: try to eliminate columns
    # A column r can be removed if its rank-1 tensor contribution is in the span
    # of the remaining columns' tensor contributions.
    # 
    # Efficient approach: For each candidate column r, check if removing it
    # still gives a valid decomposition. This is O(rank * n*m*p) per check.
    # 
    # Better: use the fact that if column r's tensor T_r = u_r ⊗ v_r ⊗ w_r
    # is in the span of {T_s : s != r}, then we can remove it.
    # 
    # Practical heuristic: try removing columns one at a time, starting with
    # those that have the smallest "importance" (norm of contribution).
    
    remaining = list(range(best_rank))
    
    # Compute importance score for each column
    col_importance = np.zeros(best_rank)
    for r in range(best_rank):
        # The contribution of column r to the tensor
        # Sum over all (i,j,k) of |U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]|
        # This is expensive, so use a proxy: product of norms
        col_importance[r] = (np.linalg.norm(U[:, r]) * 
                           np.linalg.norm(V[:, r]) * 
                           np.linalg.norm(W[:, r]))
    
    # Try to eliminate columns, starting with least important
    order = np.argsort(col_importance)
    elim_attempts = 0
    max_elim_attempts = 50
    
    for r_idx in order:
        if time.time() > deadline - 15:
            break
        if elim_attempts >= max_elim_attempts:
            break
        
        r = r_idx
        if r not in remaining:
            continue
        
        # Try removing column r
        test_remaining = [c for c in remaining if c != r]
        if not test_remaining:
            break
        
        test_U = best_U[:, test_remaining]
        test_V = best_V[:, test_remaining]
        test_W = best_W[:, test_remaining]
        
        if verify(test_U, test_V, test_W, tol=1e-8):
            remaining = test_remaining
            elim_attempts += 1
        
        elim_attempts += 1
    
    if len(remaining) < best_rank:
        final_U = best_U[:, remaining]
        final_V = best_V[:, remaining]
        final_W = best_W[:, remaining]
        
        if verify(final_U, final_V, final_W):
            best_U, best_V, best_W = final_U, final_V, final_W
            best_rank = len(remaining)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
