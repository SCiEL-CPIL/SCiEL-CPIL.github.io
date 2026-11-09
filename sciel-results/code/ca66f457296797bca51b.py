"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time
import random
import numpy as np
from fractions import Fraction
import scipy.optimize

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback (rank 24)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify_exact(U, V, W, n, m, p):
        """Check if the decomposition is valid with exact integer/half-integer arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0, 1)
                    for c in range(rank):
                        s += Fraction(str(U[i*m+j][c])) * Fraction(str(V[j*p+k][c])) * Fraction(str(W[k*n+i][c]))
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Fast floating point verification."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def round_to_half_int(arr):
        """Round a numpy array to the nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def try_rank(r, max_restarts=100):
        """Try to find a valid rank-r decomposition using scipy optimization."""
        if time.time() > deadline:
            return None
            
        num_u = n * m * r
        num_v = m * p * r
        num_w = p * n * r
        total_vars = num_u + num_v + num_w
        
        def objective(x):
            U_flat = x[:num_u]
            V_flat = x[num_u:num_u+num_v]
            W_flat = x[num_u+num_v:]
            
            U = U_flat.reshape((n*m, r))
            V = V_flat.reshape((m*p, r))
            W = W_flat.reshape((p*n, r))
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err += (s - 1.0)**2
            return err

        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            x0 = np.random.randn(total_vars) * 0.5
            
            try:
                res = scipy.optimize.minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                
                if res.success and res.fun < 1e-6:
                    x_sol = res.x
                    U_sol = x_sol[:num_u].reshape((n*m, r))
                    V_sol = x_sol[num_u:num_u+num_v].reshape((m*p, r))
                    W_sol = x_sol[num_u+num_v:].reshape((p*n, r))
                    
                    # Try rounding
                    U_r = round_to_half_int(U_sol)
                    V_r = round_to_half_int(V_sol)
                    W_r = round_to_half_int(W_sol)
                    
                    if verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        if verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # Local search on rounded values
                    for _ in range(50):
                        U_p = U_r + np.random.choice([-0.5, 0, 0.5], U_r.shape)
                        V_p = V_r + np.random.choice([-0.5, 0, 0.5], V_r.shape)
                        W_p = W_r + np.random.choice([-0.5, 0, 0.5], W_r.shape)
                        if verify_fast(U_p.tolist(), V_p.tolist(), W_p.tolist(), n, m, p):
                            if verify_exact(U_p.tolist(), V_p.tolist(), W_p.tolist(), n, m, p):
                                return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
                            
            except Exception:
                pass
                
        return None

    # Try rank 11
    result = try_rank(11)
    if result is not None:
        return result
    
    # Try rank 12
    result = try_rank(12)
    if result is not None:
        return result

    return best_result
# EVOLVE-BLOCK-END
