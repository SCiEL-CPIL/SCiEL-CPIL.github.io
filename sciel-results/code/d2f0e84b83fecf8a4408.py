import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T
    # T[u, v, w] = 1 if u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Flatten T for easier handling
    # T_flat shape: (n*m, m*p, p*n)
    
    schoolbook_rank = n * m * p
    min_rank = max(n * p, m * p, n * m)
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    deadline = 145.0 # 5s margin

    def check_solution(U, V, W):
        # Verify exact decomposition
        # T_calc[u, v, w] = sum_t U[u,t] V[v,t] W[w,t]
        # We can compute this efficiently
        # T_calc = np.einsum('ut,vt,wt->uvw', U, V, W)
        # But einsum might be slow for large r? No, r is small.
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        err = np.linalg.norm(T_calc - T)
        return err < 1e-9

    def snap_to_half_int(U, V, W):
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        return U_int, V_int, W_int

    def solve_als(r, max_iters=100, restarts=10):
        for restart in range(restarts):
            if time.time() > deadline:
                return None
                
            # Initialize
            # Use random integers or small reals
            # Try to start with structured sparsity?
            U = np.random.uniform(-1, 1, (n * m, r))
            V = np.random.uniform(-1, 1, (m * p, r))
            W = np.random.uniform(-1, 1, (p * n, r))
            
            # Normalize to prevent divergence
            # U /= np.linalg.norm(U, axis=0, keepdims=True)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None

                # Update W
                # A_UV: (n*m, m*p, r) -> (n*m*m*p, r)
                # A_UV[(u,v), t] = U[u,t] V[v,t]
                # Vectorized:
                # U[:, np.newaxis, :] has shape (nm, 1, r)
                # V[np.newaxis, :, :] has shape (1, mp, r)
                # Product has shape (nm, mp, r)
                A_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
                A_UV_flat = A_UV.reshape(n * m * m * p, r)
                
                # T_w: for each w, we need T[:, :, w] flattened
                # T shape (nm, mp, pn)
                # We want a matrix T_W of shape (nm*mp, pn) where T_W[idx, w] = T[u,v,w]
                # T.transpose(0, 1, 2) is same as T.
                # T.reshape(nm*mp, pn)
                T_W = T.reshape(n * m * m * p, p * n)
                
                # Solve A_UV_flat W = T_W
                # W has shape (pn, r)
                # We are solving for W.T? No.
                # Equation: sum_t A_UV_flat[idx, t] * W[w, t] = T_W[idx, w]
                # This is A_UV_flat @ W.T = T_W
                # So W.T = lstsq(A_UV_flat, T_W)
                # W.T shape (r, pn)
                try:
                    W_T, residuals, rank, sv = np.linalg.lstsq(A_UV_flat, T_W, rcond=None)
                    W = W_T.T
                except:
                    continue

                # Update U
                # A_VW: (m*p, p*n, r) -> (m*p*p*n, r)
                A_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
                A_VW_flat = A_VW.reshape(m * p * p * n, r)
                
                # T_u: for each u, we need T[u, :, :] flattened
                # T shape (nm, mp, pn)
                # We want T_U of shape (mp*pn, nm) where T_U[idx, u] = T[u, v, w]
                # T.transpose(1, 2, 0).reshape(mp*pn, nm)
                T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
                
                # A_VW_flat @ U.T = T_U
                try:
                    U_T, residuals, rank, sv = np.linalg.lstsq(A_VW_flat, T_U, rcond=None)
                    U = U_T.T
                except:
                    continue

                # Update V
                # A_UW: (n*m, p*n, r) -> (n*m*p*n, r)
                A_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
                A_UW_flat = A_UW.reshape(n * m * p * n, r)
                
                # T_v: for each v, we need T[:, v, :] flattened
                # T shape (nm, mp, pn)
                # We want T_V of shape (nm*pn, mp) where T_V[idx, v] = T[u, v, w]
                # T.transpose(0, 2, 1).reshape(nm*pn, mp)
                T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
                
                # A_UW_flat @ V.T = T_V
                try:
                    V_T, residuals, rank, sv = np.linalg.lstsq(A_UW_flat, T_V, rcond=None)
                    V = V_T.T
                except:
                    continue

                # Check error
                # T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                # err = np.linalg.norm(T_calc - T)
                # To save time, check every few iterations or if improvement stalls
                if it % 5 == 0:
                    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_calc - T)
                    if err < 1e-7:
                        # Try snapping
                        U_int, V_int, W_int = snap_to_half_int(U, V, W)
                        if check_solution(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                        # Also check if original float solution is "close enough" to be considered valid?
                        # The prompt requires integer/half-integer. So we must snap.
                        # If snapping fails, maybe try snapping to integers?
                        U_int2 = np.round(U)
                        V_int2 = np.round(V)
                        W_int2 = np.round(W)
                        if check_solution(U_int2, V_int2, W_int2):
                            return U_int2, V_int2, W_int2
                        # If float solution is very good, but snapping fails, 
                        # it might be a local minimum not on the lattice.
                        # We continue searching.
                        if err < 1e-9:
                            # It's exact in float, but not lattice.
                            # This shouldn't happen if the problem has a lattice solution.
                            pass

        return None

    # Try ranks
    for r in range(min_rank, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Increase restarts for smaller ranks as they are harder
        restarts = 20 if r < 40 else 10
        result = solve_als(r, max_iters=50, restarts=restarts)
        if result:
            U_int, V_int, W_int = result
            # Convert to list of lists
            U_list = U_int.tolist()
            V_list = V_int.tolist()
            W_list = W_int.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
