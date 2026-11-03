"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Build the rank-55 Strassen-based decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the true tensor
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        if time.time() > deadline - 5:
            return False
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for c in range(U_mat.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U_mat[:, c], V_mat[:, c], W_mat[:, c])
        return np.allclose(T_rec, T_true, atol=tol)
    
    def check_dependence_vectorized(U_mat, V_mat, W_mat, c):
        other_cols = [i for i in range(U_mat.shape[1]) if i != c]
        if len(other_cols) < 2:
            return False, None
        
        other_U = U_mat[:, other_cols]
        other_V = V_mat[:, other_cols]
        other_W = W_mat[:, other_cols]
        
        A_tensor = np.einsum('io,jo,ko->ijk o', other_U, other_V, other_W)
        A_mat = A_tensor.reshape(-1, len(other_cols))
        
        b_tensor = np.einsum('i,j,k->ijk', U_mat[:, c], V_mat[:, c], W_mat[:, c])
        b_vec = b_tensor.reshape(-1)
        
        b_norm = np.linalg.norm(b_vec)
        if b_norm < 1e-10:
            return False, None
        
        try:
            lambda_sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except Exception:
            return False, None
        
        if residuals is not None and len(residuals) > 0:
            residual_norm = np.sqrt(residuals[0])
            if residual_norm / b_norm < 1e-8:
                return True, lambda_sol
        return False, None
    
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    # Iterative peeling - use fast residual for checking
    peeled = True
    while peeled and time.time() < deadline - 15:
        peeled = False
        for c in range(current_rank):
            if time.time() > deadline - 15:
                break
            try:
                is_dep, _ = check_dependence_vectorized(current_U, current_V, current_W, c)
                if is_dep:
                    current_U = np.delete(current_U, c, axis=1)
                    current_V = np.delete(current_V, c, axis=1)
                    current_W = np.delete(current_W, c, axis=1)
                    current_rank -= 1
                    peeled = True
                    break
            except Exception:
                continue
    
    if current_rank < rank:
        if verify_decomposition(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
    
    # ALS-based rank reduction attempt
    # Precompute the sparse structure of T_true for fast residual evaluation
    # T_true has exactly n*m*p = 60 non-zero entries (all 1s)
    true_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_indices.append((i * m + j, j * p + k, k * n + i))
    true_indices = np.array(true_indices)
    
    def fast_residual(U_mat, V_mat, W_mat):
        """Compute residual at only the non-zero entries of T_true."""
        r = U_mat[true_indices[:, 0]]  # (nnz, rank)
        s = V_mat[true_indices[:, 1]]  # (nnz, rank)
        t = W_mat[true_indices[:, 2]]  # (nnz, rank)
        # For each entry, residual = sum_c U[i,c]*V[j,c]*W[k,c] - 1
        contrib = (r * s * t).sum(axis=1)
        return np.linalg.norm(contrib - 1.0)
    
    best_U = current_U.copy()
    best_V = current_V.copy()
    best_W = current_W.copy()
    best_rank = current_rank
    
    rng = np.random.RandomState(42)
    
    for target_rank in range(best_rank - 1, max(best_rank - 8, 40), -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple initializations
        for init_attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            if init_attempt == 0:
                # Keep the first target_rank columns
                U_init = current_U[:, :target_rank].copy()
                V_init = current_V[:, :target_rank].copy()
                W_init = current_W[:, :target_rank].copy()
            else:
                # Random linear combination of existing columns
                # This can escape local minima
                U_init = current_U @ rng.randn(current_rank, target_rank)
                V_init = current_V @ rng.randn(current_rank, target_rank)
                W_init = current_W @ rng.randn(current_rank, target_rank)
                # Scale to maintain similar magnitudes
                U_init /= np.linalg.norm(U_init) * np.sqrt(target_rank / current_rank) + 1e-10
                V_init /= np.linalg.norm(V_init) * np.sqrt(target_rank / current_rank) + 1e-10
                W_init /= np.linalg.norm(W_init) * np.sqrt(target_rank / current_rank) + 1e-10
            
            # ALS iterations
            for iteration in range(80):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each row i of U
                # Precompute VW matrix: shape (target_rank, dim_v * dim_w)
                VW = np.einsum('jc,kc->jkc', V_init, W_init).reshape(target_rank, -1)
                VW_T = VW.T  # (dim_v * dim_w, target_rank)
                
                for i in range(dim_u):
                    if time.time() > deadline - 20:
                        break
                    t_i = T_true[i, :, :].reshape(-1)
                    try:
                        U_init[i, :], _, _, _ = np.linalg.lstsq(VW_T, t_i, rcond=None)
                    except Exception:
                        pass
                
                # Update V: for each row j of V
                UW = np.einsum('ic,kc->ikc', U_init, W_init).reshape(-1, target_rank)
                UW_T = UW.T  # (dim_u * dim_w, target_rank)
                
                for j in range(dim_v):
                    if time.time() > deadline - 20:
                        break
                    t_j = T_true[:, j, :].reshape(-1)
                    try:
                        V_init[j, :], _, _, _ = np.linalg.lstsq(UW_T, t_j, rcond=None)
                    except Exception:
                        pass
                
                # Update W: for each row k of W
                UV = np.einsum('ic,jc->ijc', U_init, V_init).reshape(-1, target_rank)
                UV_T = UV.T  # (dim_u * dim_v, target_rank)
                
                for k in range(dim_w):
                    if time.time() > deadline - 20:
                        break
                    t_k = T_true[:, :, k].reshape(-1)
                    try:
                        W_init[k, :], _, _, _ = np.linalg.lstsq(UV_T, t_k, rcond=None)
                    except Exception:
                        pass
                
                # Check convergence using fast residual
                if iteration % 5 == 0:
                    res = fast_residual(U_init, V_init, W_init)
                    if res < 1e-6:
                        break
            
            # Final check with fast residual
            res = fast_residual(U_init, V_init, W_init)
            if res < 1e-6:
                # Full verification
                T_rec = np.zeros((dim_u, dim_v, dim_w))
                for c in range(target_rank):
                    T_rec += np.einsum('i,j,k->ijk', U_init[:, c], V_init[:, c], W_init[:, c])
                if np.allclose(T_rec, T_true, atol=1e-8):
                    best_U = U_init
                    best_V = V_init
                    best_W = W_init
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    # Final verification and return
    if best_rank < rank:
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for c in range(best_rank):
            T_rec += np.einsum('i,j,k->ijk', best_U[:, c], best_V[:, c], best_W[:, c])
        if np.allclose(T_rec, T_true, atol=1e-8):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    if verify_decomposition(U_np, V_np, W_np):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col_s = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col_s] = 1
                V_school[j * p + k][col_s] = 1
                W_school[k * n + i][col_s] = 1
                col_s += 1
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
