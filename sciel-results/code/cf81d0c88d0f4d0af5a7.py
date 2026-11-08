import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Direct rank-24 construction for 2x4 * 4x4 matrix multiplication
    # 
    # Key insight: We can use a hybrid approach combining Strassen-style
    # grouping with direct terms to achieve rank 24.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Use the following decomposition with rank 24:
    # For each k in {0,1,2,3}, we need C[0,k] and C[1,k]
    #
    # Strategy: Group B columns and use Karatsuba-like identities
    
    rank = 24
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Term 0-3: Direct products for B[0,k] terms
    # C[i,k] += A[i,0] * B[0,k]
    for k in range(4):
        U[u_idx(0, 0), k] = 1
        U[u_idx(1, 0), k] = 1
        V[v_idx(0, k), k] = 1
        W[w_idx(k, 0), k] = 1
        W[w_idx(k, 1), k] = 1
    
    # Term 4-7: Direct products for B[1,k] terms  
    # C[i,k] += A[i,1] * B[1,k]
    for k in range(4):
        col = 4 + k
        U[u_idx(0, 1), col] = 1
        U[u_idx(1, 1), col] = 1
        V[v_idx(1, k), col] = 1
        W[w_idx(k, 0), col] = 1
        W[w_idx(k, 1), col] = 1
    
    # Term 8-11: Strassen-style for B[2,k], B[3,k]
    # Use (A[i,2]+A[i,3]) * (B[2,k]+B[3,k]) - A[i,2]*B[2,k] - A[i,3]*B[3,k]
    # But we need to be more careful. Let's use:
    # M_k = (A[0,2]+A[0,3]) * (B[2,k]+B[3,k])
    # N_k = (A[1,2]+A[1,3]) * (B[2,k]+B[3,k])
    # Then C[0,k] += M_k - A[0,2]*B[2,k] - A[0,3]*B[3,k]
    # This doesn't save multiplications directly.
    
    # Better: Use the 2x2 Strassen on the bottom half
    # For B rows 2-3 and all 4 columns, treat as two 2x2 blocks
    
    # Actually, let's just use the proven Strassen rank-28 and verify it works
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                U[u_idx(r, a_col_off * 2 + c), col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                V[v_idx(b_row_off * 2 + r, b_col_off * 2 + c), col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                W[w_idx(c_col_off * 2 + c, r), col] += coeff
    
    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    # Verify
    def check(U_mat, V_mat, W_mat, r):
        if U_mat.shape != (n*m, r) or V_mat.shape != (m*p, r) or W_mat.shape != (p*n, r):
            return False
        if not np.allclose(2*U_mat, np.round(2*U_mat), atol=1e-5): return False
        if not np.allclose(2*V_mat, np.round(2*V_mat), atol=1e-5): return False
        if not np.allclose(2*W_mat, np.round(2*W_mat), atol=1e-5): return False
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C = A @ B
            Cf = np.zeros((n, p))
            for ii in range(n):
                for kk in range(p):
                    acc = 0.0
                    for jj in range(m):
                        acc += np.dot(U_mat[ii*m+jj, :] * V_mat[jj*p+kk, :], W_mat[kk*n+ii, :])
                    Cf[ii, kk] = acc
            if not np.allclose(C, Cf, atol=1e-5):
                return False
        return True
    
    if check(U, V, W, rank):
        U_int = np.round(2 * U) / 2.0
        V_int = np.round(2 * V) / 2.0
        W_int = np.round(2 * W) / 2.0
        if check(U_int, V_int, W_int, rank):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
