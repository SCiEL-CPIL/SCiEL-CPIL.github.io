"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target multiplication tensor T[u, v, w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def check_error(U, V, W):
        """Compute ||sum_r U[u,r]V[v,r]W[w,r] - T||"""
        # Efficient: for each r, outer product contribution
        # Rec = sum_r U[:,r] V[:,r]^T W[:,r]^T ... 
        # Use einsum: Rec[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
        Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(Rec - T)
    
    def solve_one_factor(fixed1, fixed2, target_slice, dim_size):
        """
        Solve for one factor given the other two.
        If solving for U: fixed1=V, fixed2=W, target_slice = T[u,:,:] for each u
        The equation is: sum_r U[u,r] * (V[v,r]*W[w,r]) = T[u,v,w]
        For fixed u: A @ U[u,:] = b where A[i,r] = V[v_i,r]*W[w_i,r]
        """
        R = fixed1.shape[1]
        # Precompute the product matrix P[v, w, r] = fixed1[v,r] * fixed2[w,r]
        P = fixed1[:, np.newaxis, :] * fixed2[np.newaxis, :, :]  # (dim1, dim2, R)
        A = P.reshape(dim1 * dim2, R)
        
        result = np.zeros((dim_size, R), dtype=np.float64)
        for idx in range(dim_size):
            b = target_slice[idx].reshape(-1)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                result[idx] = sol
            except np.linalg.LinAlgError:
                result[idx] = 0.0
        return result
    
    # Try to find a rank-44 decomposition
    deadline = time.time() + 140  # 10s safety margin
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    found_rank = None
    
    # Try rank 44 first (known minimal)
    for R in [44]:
        max_restarts = 30
        max_iter = 50
        tol = 1e-7
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with reasonable scaling
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update W: for each w, solve A_w @ W[w,:] = T[:,:,w].reshape(-1)
                # A_w[u*mp+v, r] = U[u,r] * V[v,r]
                P_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, R)
                A_W = P_UV.reshape(nm * mp, R)
                
                W_new = np.zeros((pn, R), dtype=np.float64)
                for w in range(pn):
                    b = T[:, :, w].reshape(-1)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                        W_new[w] = sol
                    except np.linalg.LinAlgError:
                        W_new[w] = 0.0
                W = W_new
                
                # Update V: for each v, solve A_v @ V[v,:] = T[:,v,:].reshape(-1)
                # A_v[u*pn+w, r] = U[u,r] * W[w,r]
                P_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, R)
                A_V = P_UW.reshape(nm * pn, R)
                
                V_new = np.zeros((mp, R), dtype=np.float64)
                for v in range(mp):
                    b = T[:, v, :].reshape(-1)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                        V_new[v] = sol
                    except np.linalg.LinAlgError:
                        V_new[v] = 0.0
                V = V_new
                
                # Update U: for each u, solve A_u @ U[u,:] = T[u,:,:].reshape(-1)
                # A_u[v*pn+w, r] = V[v,r] * W[w,r]
                P_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, R)
                A_U = P_VW.reshape(mp * pn, R)
                
                U_new = np.zeros((nm, R), dtype=np.float64)
                for u in range(nm):
                    b = T[u, :, :].reshape(-1)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                        U_new[u] = sol
                    except np.linalg.LinAlgError:
                        U_new[u] = 0.0
                U = U_new
                
                # Check error
                err = check_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < tol:
                    break
            
            if best_err < tol:
                break
        
        if best_err < tol:
            found_rank = R
            break
    
    # If we found a valid low-rank solution, use it
    if found_rank is not None and best_U is not None:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition of rank n*m*p = 48
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
