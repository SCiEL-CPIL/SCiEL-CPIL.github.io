import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build the 2x4x6 matrix multiplication decomposition using Strassen on 2x2x2 blocks.
    Rank = 42.
    """
    n, m, p = 2, 4, 6
    n_j_blocks = m // 2  # 2
    n_k_blocks = p // 2  # 3
    rank = n_j_blocks * n_k_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for jb in range(n_j_blocks):
        for kb in range(n_k_blocks):
            base_col = (jb * n_k_blocks + kb) * 7
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                col = base_col + c
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        U[i * m + j, col] = U_s[i * 2 + j_local, c]
                for j_local in range(2):
                    j = j0 + j_local
                    for k_local in range(2):
                        k = k0 + k_local
                        V[j * p + k, col] = V_s[j_local * 2 + k_local, c]
                for k_local in range(2):
                    k = k0 + k_local
                    for i in range(2):
                        W[k * n + i, col] = W_s[k_local * 2 + i, c]
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = 2, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error < tol, error

def _try_als_decomposition(rank, max_iter=100, tol=1e-9, seed=0):
    """
    Attempt to find a rank-`rank` decomposition using Alternating Least Squares (ALS).
    Initializes with random values scaled to match the target tensor's magnitude.
    """
    n, m, p = 2, 4, 6
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Initialize U, V, W
    np.random.seed(seed)
    # Scale initialization to avoid divergence
    scale = 1.0 / (rank ** 0.5)
    U = np.random.randn(dim_u, rank) * scale
    V = np.random.randn(dim_v, rank) * scale
    W = np.random.randn(dim_w, rank) * scale
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        if _deadline_reached():
            break
            
        # Update U
        # T is (I, J, K). We want to minimize || T - U V^T W^T ||_F^2 w.r.t U.
        # Reshape T to (I, J*K)
        T_mat = T.reshape(dim_u, dim_v * dim_w)
        # V W^T is (J, R) x (K, R)^T -> (J, K) ... wait.
        # Term r is u_r v_r w_r.
        # Sum_r u_r (v_r w_r^T)
        # For fixed V, W, let M_r = v_r w_r^T.
        # We want U * sum_r (M_r^T M_r) ? No.
        # Minimize || T - U S ||_F where S is (R, J*K) ?
        # Let's construct the "effective" matrix for V and W.
        # V is (J, R), W is (K, R).
        # The term for each r is u_r (v_r \otimes w_r).
        # Let Z be (R, J*K) where Z[r, j*K+k] = V[j,r] * W[k,r].
        # Then T ~= U Z^T.
        # So U = T Z^T (Z Z^T)^{-1} ? No, U is (I, R).
        # T (I, J*K) = U (I, R) Z^T (R, J*K).
        # So U = T (Z^T)^+ = T Z^T (Z Z^T)^{-1} ?
        # (Z Z^T) is (J*K, J*K). Z is (R, J*K).
        # Let's compute Z: (R, J*K).
        Z = np.zeros((rank, dim_v * dim_w))
        for r in range(rank):
            Z[r, :] = (V[:, r] * W[:, r]).flatten() # This is wrong. v_r is (J,1), w_r is (K,1). outer product is (J,K).
            # Correct:
            outer_vw = np.outer(V[:, r], W[:, r]) # (J, K)
            Z[r, :] = outer_vw.flatten()
        
        # Now we have T (I, J*K) and Z (R, J*K).
        # We want to find U (I, R) such that U Z^T approx T.
        # This is a linear least squares problem: U = T (Z^T)^+
        # Or solve (Z Z^T) U^T = Z T^T ?
        # Minimize || T - U Z^T ||_F^2
        # Gradient wrt U is 2 (U Z^T - T) Z = 0 => U Z^T Z = T Z
        # So U = T Z (Z^T Z)^{-1}
        
        ZtZ = Z.T @ Z # (R, R)
        TZ = T @ Z # (I, R)
        
        try:
            U = np.linalg.solve(ZtZ, TZ.T).T
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(ZtZ, TZ.T, rcond=None)[0].T
            
        # Update V
        # Minimize || T - U V W^T ||_F^2 w.r.t V.
        # Reshape T to (I*K, J) ? No.
        # Term: u_r v_r w_r.
        # For fixed U, W.
        # Let M_r = u_r w_r.
        # T (I, J, K).
        # Reshape T to (I*K, J) by moving J to the last axis?
        # T_reshaped[i*K + k, j] = T[i, j, k].
        T_v_mat = np.transpose(T, (0, 2, 1)).reshape(dim_u * dim_w, dim_v)
        
        # Z_v should be (R, I*K) such that V Z_v^T approx T_v_mat.
        # Term r: v_r (u_r w_r^T).
        # outer_uw = np.outer(U[:, r], W[:, r]) # (I, K)
        # Z_v[r, i*K+k] = U[i,r] * W[k,r]
        
        Z_v = np.zeros((rank, dim_u * dim_w))
        for r in range(rank):
            outer_uw = np.outer(U[:, r], W[:, r])
            Z_v[r, :] = outer_uw.flatten()
            
        ZtZ_v = Z_v.T @ Z_v
        TZ_v = T_v_mat @ Z_v
        
        try:
            V = np.linalg.solve(ZtZ_v, TZ_v.T).T
        except np.linalg.LinAlgError:
            V = np.linalg.lstsq(ZtZ_v, TZ_v.T, rcond=None)[0].T

        # Update W
        # Minimize || T - U V W^T ||_F^2 w.r.t W.
        # Reshape T to (I*J, K).
        T_w_mat = T.reshape(dim_u * dim_v, dim_w)
        
        # Z_w should be (R, I*J) such that W Z_w^T approx T_w_mat.
        # Term r: w_r (u_r v_r^T).
        # outer_uv = np.outer(U[:, r], V[:, r]) # (I, J)
        
        Z_w = np.zeros((rank, dim_u * dim_v))
        for r in range(rank):
            outer_uv = np.outer(U[:, r], V[:, r])
            Z_w[r, :] = outer_uv.flatten()
            
        ZtZ_w = Z_w.T @ Z_w
        TZ_w = T_w_mat @ Z_w
        
        try:
            W = np.linalg.solve(ZtZ_w, TZ_w.T).T
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(ZtZ_w, TZ_w.T, rcond=None)[0].T

        # Check convergence
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        error = np.linalg.norm(T - recon)
        
        if abs(prev_error - error) < tol * (prev_error + 1e-12):
            break
        prev_error = error
        
    # Final check
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    error = np.linalg.norm(T - recon)
    
    if error < 1e-9:
        return U, V, W, error
    else:
        return None, None, None, error

def _try_homotopy_continuation(rank, max_iter=100, seed=0):
    """
    Attempt to find a rank-`rank` decomposition using homotopy continuation.
    Start from a known rank-(rank+1) decomposition and try to drive one entry to zero,
    then remove that component. If successful, we have a rank-`rank` decomposition.
    """
    n, m, p = 2, 4, 6
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # We need a starting point of rank `rank+1`.
    # If rank+1 == 42, use the Strassen solution.
    # If rank+1 < 42, we would need a lower rank solution to start from, which we don't have.
    # So this method is only useful if we can reduce from 42 to 41.
    
    if rank + 1 != 42:
        return None, None, None, float('inf')

    U, V, W = _build_2x4x6_decomposition()
    
    # We want to try to zero out a specific component r0.
    # Let's try each component r0 in 0..41.
    for r0 in range(42):
        if _deadline_reached():
            break
            
        # We want to solve for U, V, W of size (dim, rank+1) such that the r0-th column is zero.
        # This is a constrained optimization problem.
        # Alternatively, we can fix the r0-th column to be very small and see if the rest converges to a valid solution.
        # Or, we can use the "removal" technique:
        # 1. Take the 42-rank solution.
        # 2. Pick r0.
        # 3. Add a small random perturbation to U, V, W.
        # 4. Run ALS for rank 42, but penalize the norm of the r0-th column.
        #    Actually, ALS doesn't support penalties directly.
        #    
        # Let's try a different homotopy:
        # Minimize || T - U V^T W^T ||^2 + lambda * || U[:, r0] ||^2 + || V[:, r0] ||^2 + || W[:, r0] ||^2
        # As lambda -> inf, the r0-th term goes to 0.
        # Then we remove it.
        
        # We can implement this by modifying the ALS update steps to include the regularization term.
        # This is getting complex. Let's stick to a simpler heuristic.
        
        # Heuristic: Randomly perturb the 42-rank solution and run ALS for rank 41.
        # We already do this in the main loop.
        pass
        
    return None, None, None, float('inf')

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the guaranteed Rank 42 solution
    U42, V42, W42 = _build_2x4x6_decomposition()
    
    # 2. Verify Rank 42
    is_valid_42, _ = _verify_decomposition(U42, V42, W42)
    
    if not is_valid_42:
        # Fallback to Schoolbook Rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 3. Try to find a lower rank decomposition using ALS
    # We try ranks 41, 40, 39...
    # Given the small size, we can afford a few iterations.
    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42
    
    # Try rank 41
    for target_rank in [41, 40]:
        if _deadline_reached():
            break
            
        # Try a few random seeds
        for seed in range(5):
            if _deadline_reached():
                break
                
            U_try, V_try, W_try, err = _try_als_decomposition(target_rank, max_iter=50, seed=seed)
            if U_try is not None:
                # Verify with high precision
                is_valid, _ = _verify_decomposition(U_try, V_try, W_try)
                if is_valid:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    break # Found valid rank, stop trying this rank
            
    # 4. Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
