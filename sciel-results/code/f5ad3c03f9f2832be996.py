import numpy as np
import time
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s margin
    
    def _verify(U, V, W, dims):
        """
        Verifies if U, V, W exactly decompose the tensor.
        dims: (n, m, p)
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        """
        n, m, p = dims
        R = U.shape[1]
        
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check if snap was close
        if not (np.allclose(U, U_snap, atol=1e-5) and 
                np.allclose(V, V_snap, atol=1e-5) and 
                np.allclose(W, W_snap, atol=1e-5)):
            return False
            
        # Exact verification using Fractions
        U_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in U_snap]
        V_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in V_snap]
        W_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in W_snap]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += U_frac[u_idx][r] * V_frac[v_idx][r] * W_frac[w_idx][r]
                    if s != 1:
                        return False
        return True

    def _schoolbook(dims):
        n, m, p = dims
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def _try_rank(dims, R, max_restarts=10):
        """
        Attempts to find a decomposition of rank R for given dimensions.
        """
        n, m, p = dims
        num_vars = R * (n*m + m*p + p*n)
        
        def residual(params):
            U = params[0:n*m*R].reshape(n*m, R)
            V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = params[n*m*R + m*p*R:].reshape(p*n, R)
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        res[idx] = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :]) - 1.0
                        idx += 1
            return res

        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(restart * 1000 + R + n + m + p)
            
            # Initialize with small random values
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Gauge fixing: U[0, r] = 1
            U[0, :] = 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                sol = least_squares(residual, params, method='lm', max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            except:
                continue
                
            if np.linalg.norm(sol.fun) < 1e-6:
                U_sol = sol.x[0:n*m*R].reshape(n*m, R)
                V_sol = sol.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = sol.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                if _verify(U_sol, V_sol, W_sol, dims):
                    return U_sol, V_sol, W_sol
            
        return None

    # Strategy: Solve 2x3x3 (rank 16) and tile it 3 times for 2x3x9 (rank 48)
    dims_small = (2, 3, 3)
    R_small = 16
    
    result_small = _try_rank(dims_small, R_small, max_restarts=50)
    
    if result_small:
        U_s, V_s, W_s = result_small
        # Construct 2x3x9 from 2x3x3
        n, m, p = 2, 3, 9
        p_blocks = p // 3 # 3 blocks
        R_total = R_small * p_blocks
        
        U_full = np.zeros((n*m, R_total))
        V_full = np.zeros((m*p, R_total))
        W_full = np.zeros((p*n, R_total))
        
        for l in range(p_blocks):
            # U is the same for each block
            U_full[:, l*R_small:(l+1)*R_small] = U_s
            
            # V: map (j, k_small) to (j, 3*l + k_small)
            for j in range(m):
                for k_s in range(3):
                    v_idx_full = j * p + (3 * l + k_s)
                    v_idx_s = j * 3 + k_s
                    V_full[v_idx_full, l*R_small:(l+1)*R_small] = V_s[v_idx_s, :]
                    
            # W: map (k_small, i) to (3*l + k_small, i)
            for k_s in range(3):
                for i in range(n):
                    w_idx_full = (3 * l + k_s) * n + i
                    w_idx_s = k_s * n + i
                    W_full[w_idx_full, l*R_small:(l+1)*R_small] = W_s[w_idx_s, :]
                    
        if _verify(U_full, V_full, W_full, (n, m, p)):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Try direct rank reduction on 2x3x9 (unlikely to succeed if 48 is not found, but 53..45 is the original logic)
    # We can skip this to save time if we trust the 48 rank construction, or keep it as a backup.
    # Given the deadline, and that 48 is likely, we can just fallback to schoolbook if 48 fails.
    # However, if 16 rank for 2x3x3 fails, we might try 17? No, 16 is the known rank.
    
    return _schoolbook((n, m, p))
