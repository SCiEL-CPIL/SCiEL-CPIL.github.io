import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build the target multiplication tensor T[i,j,k] = delta_{i,k}
    # We'll work with the factor representation directly.

    # --- Rank-50 block Strassen decomposition ---
    # Split inner dimension m=4 into two blocks of size 2: J1={0,1}, J2={2,3}
    # Split output dimension p=7 into three blocks: K1={0,1}, K2={2,3}, K3={4,5,6}
    # Input rows n=2: I={0,1}
    #
    # For each (I-block, J-block, K-block) triple, we multiply a 2x2 * 2x2 or 2x2 * 2x3
    # Use Strassen for 2x2*2x2 (rank 7), and a 2x2*2x3 method (rank 9)
    #
    # Actually, let's use a simpler approach: 
    # For 2x2 * 2x2: Strassen rank 7
    # For 2x2 * 2x3: we can pad to 2x2*2x3, rank 9 (or use 3 Strassen 2x2's with sharing)
    #
    # Total: 2 (I-blocks) * 2 (J-blocks) * [2 (K-blocks of size 2) * 7 + 1 (K-block of size 3) * 9]
    # = 2 * 2 * [2*7 + 9] = 2*2*23 = 92. That's worse!
    #
    # Better: don't split I. Just split J and K.
    # For each J-block (size 2) and K-block:
    #   J1-K1: 2x2*2x2 -> rank 7
    #   J1-K2: 2x2*2x2 -> rank 7  
    #   J1-K3: 2x2*2x3 -> rank 9
    #   J2-K1: 2x2*2x2 -> rank 7
    #   J2-K2: 2x2*2x2 -> rank 7
    #   J2-K3: 2x2*2x3 -> rank 9
    # Total: 7+7+9+7+7+9 = 46. But these share the I dimension (n=2), so we can't just add.
    #
    # The issue: the output C[i,k] sums over ALL j. So we need to combine contributions from J1 and J2.
    #
    # Let me think differently. Use the standard approach:
    # C = A * B where A is 2x4, B is 4x7.
    # Split B into B1 (4x2) and B2 (4x2) and B3 (4x3).
    # C1 = A*B1 (2x2), C2 = A*B2 (2x2), C3 = A*B3 (2x3).
    # For C1 and C2, use Strassen on the 2x4 * 4x2 by splitting A into A1(2x2), A2(2x2).
    # C1 = A1*B1a + A2*B1b where B1a is 2x2, B1b is 2x2.
    # Strassen: C1 = A1*B1a + A2*B1b can be done in rank 7 if we view it as 2x2*2x2+2x2*2x2.
    # But that's two separate 2x2 multiplications, each rank 7, total 14.
    #
    # I think the rank-50 comes from a specific clever construction. Let me just implement
    # a general tensor rank optimization and hope it finds something.

    # Alternative structural change: use **randomized SVD-based rank approximation** 
    # followed by **exact rational recovery**.
    
    # Build the tensor as a 3D array and try to find low-rank decompositions
    # using the fact that for matrix multiplication, the tensor has special structure.
    
    # Let's try: for each rank r, randomly initialize and use ALS (alternating least squares)
    # with a penalty for non-half-integer values, then check exactness.
    
    def _als_rank(r, n_iter=200, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for k in range(p):
                if i == k:
                    for j in range(m):
                        T[i, j, k] = 1.0
        
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # recon[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                    # For fixed V,W: recon[i,j,:] = U[i*m+j,:] @ M_ij where M_ij[k,t] = V[j*p+k,t]*W[k*n+i,t]
                    M_ij = np.zeros((p, r))
                    for k in range(p):
                        for t in range(r):
                            M_ij[k, t] = V[j*p+k, t] * W[k*n+i, t]
                    # Solve: M_ij^T @ U[i*m+j,:] = T[i,j,:]
                    A_mat = M_ij.T  # (r, p)
                    b_vec = T[i, j, :]  # (p,)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[i*m+j, :] = sol
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M_jk = np.zeros((n, r))
                    for i in range(n):
                        for t in range(r):
                            M_jk[i, t] = U[i*m+j, t] * W[k*n+i, t]
                    A_mat = M_jk.T
                    b_vec = T[:, j, k]
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[j*p+k, :] = sol
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M_ki = np.zeros((m, r))
                    for j in range(m):
                        for t in range(r):
                            M_ki[j, t] = U[i*m+j, t] * V[j*p+k, t]
                    A_mat = M_ki.T
                    b_vec = T[i, :, k]
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[k*n+i, :] = sol
        
        # Check exactness with rounding
        def _check(U, V, W):
            recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i,j,k] = sum(U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] for t in range(r))
            return np.max(np.abs(recon - T))
        
        err = _check(U, V, W)
        if err < 1e-6:
            Ur = np.round(U*2)/2
            Vr = np.round(V*2)/2
            Wr = np.round(W*2)/2
            err2 = _check(Ur, Vr, Wr)
            if err2 < 1e-9:
                return Ur.tolist(), Vr.tolist(), Wr.tolist()
        return None
    
    # Schoolbook fallback
    rank_school = n*m*p
    U_s = [[0]*rank_school for _ in range(n*m)]
    V_s = [[0]*rank_school for _ in range(m*p)]
    W_s = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    best_result = (U_s, V_s, W_s)
    best_rank = rank_school
    
    # Try ranks from 48 down to 40
    for r in range(48, 39, -1):
        if time.time() > deadline - 20:
            break
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            result = _als_rank(r, n_iter=150, seed=42+r*100+seed)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = r
                break
        if best_rank < rank_school:
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
