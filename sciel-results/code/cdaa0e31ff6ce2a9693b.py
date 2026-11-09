import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 130  # 130s safety margin before 150s kill
    
    # Schoolbook rank-36 decomposition as baseline
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p
    
    # Verify decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    def verify(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by optimizing
    # Strategy: use alternating linear solves with multiple random initializations
    
    def try_rank(r, max_iter=100, n_restarts=5):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline - 10:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random or structured initialization
            if restart == 0:
                # Use first r columns of schoolbook
                U = U0[:, :r].copy().astype(float)
                V = V0[:, :r].copy().astype(float)
                W = W0[:, :r].copy().astype(float)
            else:
                # Random initialization with small values
                scale = 1.0 / np.sqrt(r)
                U = np.random.randn(n * m, r) * scale
                V = np.random.randn(m * p, r) * scale
                W = np.random.randn(p * n, r) * scale
                # Add some structure
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i * m + j < r:
                                U[i*m+j, i*m+j] += 1.0
                            if j * p + k < r:
                                V[j*p+k, j*p+k] += 1.0
                            if k * n + i < r:
                                W[k*n+i, k*n+i] += 1.0
            
            def solve_W(U, V, r):
                """Given U and V, solve for W such that constraints hold."""
                A_mat = np.zeros((n*m*p, p*n*r))
                b_vec = np.ones(n*m*p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r_idx in range(r):
                                A_mat[eq_idx, (k*n+i)*r + r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                            eq_idx += 1
                
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_sol = sol.reshape(p*n, r)
                    residual = np.linalg.norm(A_mat @ sol - b_vec)
                    return W_sol, residual
                except:
                    return None, float('inf')
                
            def solve_U(V, W, r):
                """Given V and W, solve for U."""
                A_mat = np.zeros((n*m*p, n*m*r))
                b_vec = np.ones(n*m*p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r_idx in range(r):
                                A_mat[eq_idx, (i*m+j)*r + r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                            eq_idx += 1
                
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_sol = sol.reshape(n*m, r)
                    residual = np.linalg.norm(A_mat @ sol - b_vec)
                    return U_sol, residual
                except:
                    return None, float('inf')
                    
            def solve_V(U, W, r):
                """Given U and W, solve for V."""
                A_mat = np.zeros((n*m*p, m*p*r))
                b_vec = np.ones(n*m*p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r_idx in range(r):
                                A_mat[eq_idx, (j*p+k)*r + r_idx] = U[i*m+j, r_idx] * W[k*n+i, r_idx]
                            eq_idx += 1
                
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_sol = sol.reshape(m*p, r)
                    residual = np.linalg.norm(A_mat @ sol - b_vec)
                    return V_sol, residual
                except:
                    return None, float('inf')
            
            # Alternating minimization
            for iteration in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                W_new, res_w = solve_W(U, V, r)
                if W_new is not None:
                    W = W_new
                
                U_new, res_u = solve_U(V, W, r)
                if U_new is not None:
                    U = U_new
                
                V_new, res_v = solve_V(U, W, r)
                if V_new is not None:
                    V = V_new
                
                # Check convergence
                if verify(U, V, W, tol=1e-4):
                    return U, V, W
            
            # Final check
            if verify(U, V, W, tol=1e-4):
                return U, V, W
        
        return None
    
    # Try ranks from 35 down to 20
    for r in range(n*m*p - 1, 19, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = r
            # Try to go lower
            continue
        else:
            # If we can't find rank r, stop
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-4):
        # Fall back to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
