"""Bootstrap MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Strategy: Use block-Strassen (rank 189) as baseline.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses block-Strassen as baseline.
    """
    deadline = time.time() + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build block-Strassen decomposition (rank 189)
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
    
    rank_bs = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
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
    
    best_U, best_V, best_W = U_bs.copy(), V_bs.copy(), W_bs.copy()
    best_rank = rank_bs
    
    # Verify the decomposition is correct (Vectorized)
    def verify(U, V, W, tol=1e-9):
        """Check that the decomposition matches A@B for random A, B"""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        # Reshape U, V, W to (n, m, R), (m, p, R), (p, n, R)
        # U[i, j, r], V[j, k, r], W[k, i, r]
        U_reshaped = U.reshape(n, m, U.shape[1])
        V_reshaped = V.reshape(m, p, V.shape[1])
        W_reshaped = W.reshape(p, n, W.shape[1])
        
        # Compute C[i, k] = sum_r W[k, i, r] * (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k])
        # Let U_A[i, r] = sum_j U[i, j, r] A[i, j]
        # Let V_B[k, r] = sum_j V[j, k, r] B[j, k]
        
        # U_A: (n, R)
        # U[i, j, r] * A[i, j] -> sum over j
        # Use einsum: 'ijr,ij->ir'
        U_A = np.einsum('ijr,ij->ir', U_reshaped, A)
        
        # V_B: (p, R)
        # V[j, k, r] * B[j, k] -> sum over j
        # Use einsum: 'jkr,jk->kr'
        V_B = np.einsum('jkr,jk->kr', V_reshaped, B)
        
        # C[i, k] = sum_r W[k, i, r] * U_A[i, r] * V_B[k, r]
        # W[k, i, r] * U_A[i, r] * V_B[k, r] -> sum over r
        # Use einsum: 'kir,ir,kr->ik'
        C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, U_A, V_B)
        
        return np.allclose(C_computed, C_expected, atol=tol)
    
    # Verify block-Strassen
    if not verify(U_bs, V_bs, W_bs):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_schoolbook
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
