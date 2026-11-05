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

def _try_rank16_from_rank17(U17, V17, W17, T, deadline):
    """Try to find rank-16 by dropping one column and re-solving via ALS."""
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    best_result = None
    best_err = np.inf
    
    # Try dropping each column and see if we can recover with ALS
    for drop_col in range(17):
        if time.time() > deadline:
            break
        
        # Create rank-16 initialization by removing the dropped column
        keep = [c for c in range(17) if c != drop_col]
        U = U17[:, keep].copy()
        V = V17[:, keep].copy()
        W = W17[:, keep].copy()
        
        # Run alternating least squares
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Solve for U: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed V,W: T[a,:,:] = sum_r U[a,r] outer(V[:,r], W[:,r])
            # Reshape: T_mat[a, b*c] = sum_r U[a,r] (V[:,r] * W[:,r]).T.flatten()
            VW = np.einsum('br,cr->bc', V, W)  # (n_b, n_c)
            T_mat = T.reshape(n_a, n_b * n_c)
            # T_mat[a, bc] = sum_r U[a,r] * VW_reshaped[r, bc]
            VW_flat = VW.T.reshape(n_c, -1).T  # (n_b*n_c, rank) but need per-r
            # Actually: VW[r, b, c] = V[b,r]*W[c,r]
            # T_mat[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Let's build the design matrix properly
            # For each a: T[a,:,:] = sum_r U[a,r] * np.outer(V[:,r], W[:,r])
            # Vectorize: T_mat[a, bc] = sum_r U[a,r] * M[r, bc] where M[r,bc] = V[b,r]*W[c,r]
            M = np.einsum('br,cr->bcr', V, W).reshape(16, n_b * n_c)  # (rank, n_b*n_c)
            # T_mat = U @ M.T
            # Solve: U = T_mat @ M.T @ (M @ M.T)^{-1}
            try:
                MMt = M @ M.T  # (rank, rank)
                M_T = M.T  # (n_b*n_c, rank)
                rhs = T_mat @ M_T  # (n_a, rank)
                U_new = np.linalg.solve(MMt, rhs.T).T
            except np.linalg.LinAlgError:
                break
            
            # Solve for V
            UW = np.einsum('ar,cr->acr', U_new, W).reshape(16, n_a * n_c)  # (rank, n_a*n_c)
            T_mat_v = T.reshape(n_b, n_a * n_c)
            try:
                UWT = UW @ UW.T
                rhs_v = T_mat_v @ UW.T
                V_new = np.linalg.solve(UWT, rhs_v.T).T
            except np.linalg.LinAlgError:
                break
            
            # Solve for W
            UV = np.einsum('ar,br->abr', U_new, V_new).reshape(16, n_a * n_b)  # (rank, n_a*n_b)
            T_mat_w = T.reshape(n_c, n_a * n_b)
            try:
                UVT = UV @ UV.T
                rhs_w = T_mat_w @ UV.T
                W_new = np.linalg.solve(UVT, rhs_w.T).T
            except np.linalg.LinAlgError:
                break
            
            # Check convergence
            recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_result = (U_new.copy(), V_new.copy(), W_new.copy())
            
            if err < 1e-10:
                break
            
            U, V, W = U_new, V_new, W_new
    
    if best_result is not None and best_err < 1e-8:
        return best_result, best_err
    return None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
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
    
    # Try to find rank-16
    result16, err16 = _try_rank16_from_rank17(U17, V17, W17, T, deadline)
    
    if result16 is not None and err16 < 1e-8:
        U16, V16, W16 = result16
        # Verify
        if _verify_decomposition(U16, V16, W16, T):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
