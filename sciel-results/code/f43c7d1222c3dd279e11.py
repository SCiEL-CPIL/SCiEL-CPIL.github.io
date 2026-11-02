"""MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Uses block-Strassen (rank 189) as baseline, then attempts rank reduction via
linear algebra on the tensor structure.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    """
    deadline = time.time() + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Block-Strassen: 6x6 = 3x3 blocks of 2x2, using Strassen (rank 7) per 2x2
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] gives correct matrix multiplication tensor."""
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_mat_2d = U_mat[:, r].reshape(n, m)
            v_mat_2d = V_mat[:, r].reshape(m, p)
            w_mat_2d = W_mat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat_2d, v_mat_2d, w_mat_2d)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=atol)
    
    if not verify_decomposition(U_bs, V_bs, W_bs):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction using linear algebra
    # Key insight: if columns are linearly dependent in U and V simultaneously,
    # we can reduce rank. Use SVD-based approach to find minimal rank.
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=100, atol=1e-10):
        """Try to reduce rank by finding and merging linearly dependent column triples."""
        U = U_mat.copy()
        V = V_mat.copy()
        W = W_mat.copy()
        rank = U.shape[1]
        
        active = list(range(rank))
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if len(active) < 2:
                break
            
            merged = False
            for i in range(len(active)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(active)):
                    c1 = active[i]
                    c2 = active[j]
                    
                    u1 = U[:, c1]
                    u2 = U[:, c2]
                    mask_u2 = np.abs(u2) > atol
                    if not np.any(mask_u2):
                        continue
                    idx = np.where(mask_u2)[0]
                    denom = np.sum(u2[idx] * u2[idx])
                    if denom < atol:
                        continue
                    alpha = np.sum(u1[idx] * u2[idx]) / denom
                    if not np.allclose(u1, alpha * u2, atol=atol):
                        continue
                    
                    v1 = V[:, c1]
                    v2 = V[:, c2]
                    mask_v2 = np.abs(v2) > atol
                    if not np.any(mask_v2):
                        continue
                    idx_v = np.where(mask_v2)[0]
                    denom_v = np.sum(v2[idx_v] * v2[idx_v])
                    if denom_v < atol:
                        continue
                    beta = np.sum(v1[idx_v] * v2[idx_v]) / denom_v
                    if not np.allclose(v1, beta * v2, atol=atol):
                        continue
                    
                    w1 = W[:, c1]
                    w2 = W[:, c2]
                    if not np.allclose(w1, alpha * beta * w2, atol=atol):
                        continue
                    
                    W[:, c2] = W[:, c2] + w1
                    active.remove(c1)
                    merged = True
                    break
                if merged:
                    break
            
            if not merged:
                break
        
        if len(active) < rank:
            U = U[:, active]
            V = V[:, active]
            W = W[:, active]
        
        return U, V, W, len(active)
    
    U_red, V_red, W_red, new_rank = try_reduce_rank(U_bs, V_bs, W_bs)
    
    if verify_decomposition(U_red, V_red, W_red):
        return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    return {"U": U_bs.tolist(), "V": V_bs.tolist(), "W": W_bs.tolist()}
