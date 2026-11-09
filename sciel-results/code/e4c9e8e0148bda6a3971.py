"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Target Tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def compute_error(U, V, W):
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(T_recon - T)

    def try_rank(R, max_iters=100, restarts=20):
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for r in range(restarts):
            # Random init with small values to avoid divergence?
            # Or standard normal.
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            for it in range(max_iters):
                # Update U
                T_unf1 = T.reshape(n*m, -1)
                M = np.zeros((R, m*p * p*n))
                for rr in range(R):
                    M[rr, :] = np.outer(V[:, rr], W[:, rr]).flatten()
                
                A = M.T
                AA_t = A @ A.T + 1e-8 * np.eye(R)
                try:
                    U_new = (T_unf1 @ A.T) @ np.linalg.inv(AA_t)
                except:
                    U_new = U

                # Update V
                T_unf2 = T.reshape(m*p, -1)
                N = np.zeros((R, n*m * p*n))
                for rr in range(R):
                    N[rr, :] = np.outer(U[:, rr], W[:, rr]).flatten()
                A = N.T
                AA_t = A @ A.T + 1e-8 * np.eye(R)
                try:
                    V_new = (T_unf2 @ A.T) @ np.linalg.inv(AA_t)
                except:
                    V_new = V

                # Update W
                T_unf3 = T.reshape(p*n, -1)
                P = np.zeros((R, n*m * m*p))
                for rr in range(R):
                    P[rr, :] = np.outer(U[:, rr], V[:, rr]).flatten()
                A = P.T
                AA_t = A @ A.T + 1e-8 * np.eye(R)
                try:
                    W_new = (T_unf3 @ A.T) @ np.linalg.inv(AA_t)
                except:
                    W_new = W

                U, V, W = U_new, V_new, W_new
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-10:
                    break
            # End iters
            if best_err < 1e-8:
                break # Found a good solution for this rank, stop restarting?
                # No, maybe another restart finds a cleaner one?
                # But if error is 1e-10, it's likely the global min for this rank.
                
        if best_err < 1e-6:
            # Try to clean up
            # 1. Integers
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if compute_error(U_int, V_int, W_int) < 1e-8:
                return U_int, V_int, W_int
            
            # 2. Half-integers
            U_half = np.round(best_U * 2) / 2.0
            V_half = np.round(best_V * 2) / 2.0
            W_half = np.round(best_W * 2) / 2.0
            if compute_error(U_half, V_half, W_half) < 1e-8:
                return U_half, V_half, W_half
            
            # 3. Maybe the solution is close to integers but slightly off due to noise?
            # If best_err is very small, we can try to force it?
            # But if it doesn't match exactly, it's invalid.
            pass
        return None

    # Search for lower rank
    start_time = time.time()
    # Try ranks from lower bound to schoolbook
    # Lower bound: 17
    # Let's try up to 30
    for R in range(17, 31):
        if time.time() - start_time > 120:
            break
        result = try_rank(R, max_iters=50, restarts=20)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }

# EVOLVE-BLOCK-END
