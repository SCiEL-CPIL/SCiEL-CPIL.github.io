import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target tensor T flattened
    # T[i, j, k] is 1 if we are computing C[i, k] from A[i, j] and B[j, k]
    # Flattened indices:
    # a = i * m + j
    # b = j * p + k
    # c = k * n + i
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check if sum_r U[:, r] V[:, r] W[:, r] == T
        # T is (nm, mp, pn)
        # Term r is outer product U[:, r] (mp, 1) ... no.
        # Term is U[:, r] * V[:, r] * W[:, r] ? No.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        R = U.shape[1]
        # Efficient check:
        # Sum over r of (U[:, r] * V[:, r]) outer W[:, r]
        # This is a tensor contraction.
        # We can do:
        # M = (U * V) @ W.T ? No.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Let M_ab = sum_r U[a, r] V[b, r]. This is (nm, mp).
        # Then T[a, b, c] = sum_r M_ab_r W[c, r]? No.
        # M_ab is not enough because r is summed.
        # We can compute:
        # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow for large R, but R is small.
        try:
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        except Exception:
            return False, 1e10
            
        diff = np.max(np.abs(T_calc - T))
        return diff < 1e-9, diff

    def rationalize(M, tol=1e-6):
        # Round to nearest half-integer
        M_half = np.round(M * 2) / 2.0
        return M_half

    # Try to find a decomposition with lower rank
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: Schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Time budget
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    # Try ranks from 50 down to 30?
    # We will try a few specific ranks.
    for R in [50, 45, 40, 35]:
        if time.time() > deadline:
            break
            
        # Initialize
        np.random.seed(R)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # ALS iterations
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Update U
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed a, T[a, :, :] = sum_r U[a, r] (V[:, r] outer W[:, r])
            # We want to minimize || T[a, :, :] - sum_r U[a, r] M_r ||^2
            # where M_r = V[:, r] outer W[:, r]
            # This is a linear system for U[a, :].
            # Let M_mat be (mp*pn, R) where column r is V[:, r] * W[:, r] (flattened)
            M_mat = np.zeros((m * p * p * n, R))
            for r in range(R):
                M_mat[:, r] = (V[:, r] * W[:, r]).flatten()
                
            # T_flat[a] is T[a, :, :].flatten()
            T_flat = T.reshape(n * m, -1)
            
            # Solve U = T_flat @ M_mat.T @ inv(M_mat @ M_mat.T)
            # Regularize
            reg = 1e-6
            MtM = M_mat.T @ M_mat + reg * np.eye(R)
            MtT = M_mat.T @ T_flat.T # (R, nm)
            U_new = np.linalg.solve(MtM, MtT).T
            
            # Update V
            M_mat_UW = np.zeros((n * m * p * n, R))
            for r in range(R):
                M_mat_UW[:, r] = (U_new[:, r] * W[:, r]).flatten()
            V_new = np.linalg.solve(MtM, M_mat_UW.T @ T_flat.T).T # Wait, T_flat is (nm, mp*pn). 
            # V update: T[b, :, :] = sum_r V[b, r] (U[:, r] outer W[:, r])
            # T_flat is indexed by a. We need T indexed by b.
            # T is (nm, mp, pn).
            # T[:, b, :] is (nm, pn).
            # Let's reshape T to (mp, nm*pn)
            T_flat_b = T.reshape(m * p, -1)
            V_new = np.linalg.solve(MtM, M_mat_UW.T @ T_flat_b.T).T
            
            # Update W
            M_mat_UV = np.zeros((n * m * m * p, R))
            for r in range(R):
                M_mat_UV[:, r] = (U_new[:, r] * V_new[:, r]).flatten()
            T_flat_c = T.reshape(p * n, -1) # (pn, nm*mp)
            W_new = np.linalg.solve(MtM, M_mat_UV.T @ T_flat_c.T).T
            
            U, V, W = U_new, V_new, W_new
            
        # Check error
        is_valid, err = verify(U, V, W)
        if is_valid:
            # Rationalize
            U_r = rationalize(U)
            V_r = rationalize(V)
            W_r = rationalize(W)
            is_valid_r, err_r = verify(U_r, V_r, W_r)
            if is_valid_r:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            # If error is small, try rationalization anyway
            if err < 1e-4:
                U_r = rationalize(U)
                V_r = rationalize(V)
                W_r = rationalize(W)
                is_valid_r, err_r = verify(U_r, V_r, W_r)
                if is_valid_r:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # If no better solution found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
