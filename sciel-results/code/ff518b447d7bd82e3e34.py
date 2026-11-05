"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _rank17_solution(n, m, p):
    """Valid rank-17 decomposition: Strassen(2x2x2) + 10 schoolbook terms."""
    rank = 17
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Strassen for 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Schoolbook for remaining terms
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_rank16_algebraic(U17, V17, W17, T, deadline):
    """
    Try to find rank-16 by checking if one column can be expressed as a
    linear combination of the others. For each column d to drop, we check
    if there exist coefficients alpha such that:
      U[:,d] = sum_{r != d} alpha_r * U[:,r]
      V[:,d] = sum_{r != d} alpha_r * V[:,r]
      W[:,d] = sum_{r != d} alpha_r * W[:,r]
    This is a linear system in alpha.
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    rank = 17
    
    # For each column d to drop, try to express it as combo of others
    for d in range(rank):
        if time.time() > deadline:
            break
        
        # Build design matrix from all other columns
        keep = [c for c in range(rank) if c != d]
        
        # Try to find alpha such that U[:,d] = U_keep @ alpha
        U_keep = U17[:, keep]  # (n_a, 16)
        U_d = U17[:, d]  # (n_a,)
        
        # Solve U_keep @ alpha = U_d
        try:
            alpha_u, residuals_u, rank_u, sv_u = np.linalg.lstsq(U_keep, U_d, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if this alpha also works for V and W
        V_keep = V17[:, keep]  # (n_b, 16)
        V_d = V17[:, d]  # (n_b,)
        V_pred = V_keep @ alpha_u
        
        W_keep = W17[:, keep]  # (n_c, 16)
        W_d = W17[:, d]  # (n_c,)
        W_pred = W_keep @ alpha_u
        
        err_v = np.linalg.norm(V_d - V_pred)
        err_w = np.linalg.norm(W_d - W_pred)
        err_u = np.linalg.norm(U_d - U_keep @ alpha_u)
        
        if err_u < 1e-10 and err_v < 1e-10 and err_w < 1e-10:
            # Found valid rank-16 decomposition
            U16 = U17[:, keep].copy()
            V16 = V17[:, keep].copy()
            W16 = W17[:, keep].copy()
            return U16, V16, W16
    
    return None

def _try_rank16_perturbation(U17, V17, W17, T, deadline):
    """
    Try small perturbations of the rank-17 solution to find rank-16.
    We allow one column to be a linear combination with small coefficients.
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    rank = 17
    best_result = None
    best_err = np.inf
    
    # Try dropping each column and doing a least-squares fit
    for d in range(rank):
        if time.time() > deadline:
            break
        
        keep = [c for c in range(rank) if c != d]
        U_keep = U17[:, keep]
        V_keep = V17[:, keep]
        W_keep = W17[:, keep]
        
        # Try to find a rank-16 decomposition by solving for new columns
        # that best approximate the tensor
        # We'll use a simple approach: try to find U16, V16, W16 by
        # iterative refinement starting from the kept columns
        
        U = U_keep.copy()
        V = V_keep.copy()
        W = W_keep.copy()
        
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Solve for U given V, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed V,W: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            VW = np.einsum('br,cr->bcr', V, W)  # (16, n_b, n_c)
            T_mat = T.reshape(n_a, n_b * n_c)
            M = VW.reshape(16, n_b * n_c)  # (16, n_b*n_c)
            
            try:
                # T_mat = U @ M.T
                # U = T_mat @ M.T @ (M @ M.T)^{-1}
                MMt = M @ M.T
                rhs = T_mat @ M.T
                U_new = np.linalg.solve(MMt, rhs.T).T
            except np.linalg.LinAlgError:
                break
            
            # Solve for V given U, W
            UW = np.einsum('ar,cr->acr', U_new, W).reshape(16, n_a * n_c)
            T_mat_v = T.reshape(n_b, n_a * n_c)
            try:
                UWT = UW @ UW.T
                rhs_v = T_mat_v @ UW.T
                V_new = np.linalg.solve(UWT, rhs_v.T).T
            except np.linalg.LinAlgError:
                break
            
            # Solve for W given U, V
            UV = np.einsum('ar,br->abr', U_new, V_new).reshape(16, n_a * n_b)
            T_mat_w = T.reshape(n_c, n_a * n_b)
            try:
                UVT = UV @ UV.T
                rhs_w = T_mat_w @ UV.T
                W_new = np.linalg.solve(UVT, rhs_w.T).T
            except np.linalg.LinAlgError:
                break
            
            # Check error
            recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            err = np.linalg.norm(recon - T)
            
            if err < best_err:
                best_err = err
                best_result = (U_new.copy(), V_new.copy(), W_new.copy())
            
            if err < 1e-10:
                return U_new, V_new, W_new
            
            U, V, W = U_new, V_new, W_new
    
    if best_result is not None and best_err < 1e-8:
        return best_result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Get rank-17 solution
    U17, V17, W17 = _rank17_solution(n, m, p)
    T = _build_tensor(n, m, p)
    
    # Verify rank-17 is correct
    if not _verify_decomposition(U17, V17, W17, T):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try algebraic rank-16 search (fast)
    result16 = _try_rank16_algebraic(U17, V17, W17, T, deadline)
    if result16 is not None:
        U16, V16, W16 = result16
        if _verify_decomposition(U16, V16, W17, T):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Try perturbation-based rank-16 search (slower)
    result16b = _try_rank16_perturbation(U17, V17, W17, T, deadline)
    if result16b is not None:
        U16, V16, W16 = result16b
        if _verify_decomposition(U16, V16, W16, T):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
