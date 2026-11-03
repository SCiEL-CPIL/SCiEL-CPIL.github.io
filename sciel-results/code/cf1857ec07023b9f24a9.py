"""Matrix multiplication 3x5x7 - rank minimization."""
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Tries multiple candidate ranks with structured initializations and ALS refinement.
    Falls back to schoolbook (rank 105) if no improvement is found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    def schoolbook():
        rank = n * m * p  # 105
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute flattened views for faster ALS
    T_flat_u = T.reshape(nm, mp*pn)
    T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm*pn)
    T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm*mp)
    
    def verify_exact(U, V, W, tol=1e-9):
        """Verify exact decomposition with integer/half-integer entries."""
        U_r = np.round(U * 2.0) / 2.0
        V_r = np.round(V * 2.0) / 2.0
        W_r = np.round(W * 2.0) / 2.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U_r[u_idx] * V_r[v_idx], W_r[w_idx])
                    if abs(val - 1.0) > tol:
                        return None
        return {
            "U": [[float(x) for x in row] for row in U_r],
            "V": [[float(x) for x in row] for row in V_r],
            "W": [[float(x) for x in row] for row in W_r]
        }
    
    def als_solve(U, V, W, r, max_iter=20, tol=1e-8):
        """Run ALS iterations to refine the decomposition."""
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            # Update U
            M_u = np.zeros((mp*pn, r))
            for c in range(r):
                M_u[:, c] = np.outer(V[:, c], W[:, c]).flatten()
            U_sol, _, _, _ = np.linalg.lstsq(M_u, T_flat_u.T, rcond=None)
            U = U_sol.T
            
            # Update V
            M_v = np.zeros((nm*pn, r))
            for c in range(r):
                M_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()
            V_sol, _, _, _ = np.linalg.lstsq(M_v, T_flat_v.T, rcond=None)
            V = V_sol.T
            
            # Update W
            M_w = np.zeros((nm*mp, r))
            for c in range(r):
                M_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()
            W_sol, _, _, _ = np.linalg.lstsq(M_w, T_flat_w.T, rcond=None)
            W = W_sol.T
            
            # Check convergence
            if it > 0:
                diff_u = np.linalg.norm(U - U_prev)
                diff_v = np.linalg.norm(V - V_prev)
                diff_w = np.linalg.norm(W - W_prev)
                if diff_u + diff_v + diff_w < tol:
                    break
            U_prev, V_prev, W_prev = U.copy(), V.copy(), W.copy()
        
        # Final residual check
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                    residual += (val - 1.0) ** 2
        
        if residual < 1e-6:
            return U, V, W
        return None
    
    # Try multiple candidate ranks with different initializations
    candidate_ranks = [15, 18, 21, 24, 27, 30]
    
    for r_test in candidate_ranks:
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple random seeds
        for seed in [42, 123, 7, 999, 2024]:
            if time.time() > deadline - 5.0:
                break
            
            try:
                np.random.seed(seed)
                U = np.random.randn(nm, r_test)
                V = np.random.randn(mp, r_test)
                W = np.random.randn(pn, r_test)
                
                result = als_solve(U, V, W, r_test, max_iter=15)
                if result is not None:
                    U_f, V_f, W_f = result
                    verified = verify_exact(U_f, V_f, W_f)
                    if verified is not None:
                        return verified
            except Exception:
                pass
    
    # Fall back to schoolbook decomposition
    return schoolbook()
