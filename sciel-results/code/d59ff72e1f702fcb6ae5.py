import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # Indices: i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T shape: (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # We want U (n*m, r), V (m*p, r), W (p*n, r)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == T[i, j, k]
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is roughly n*m*p / (something), but 7 is for 2x2x2.
    # Let's try 10, 11, 12.
    
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    best_result = None
    
    for r in [10, 11, 12]:
        if time.time() > deadline - 20.0:
            break
            
        found = _find_decomposition(T, n, m, p, r, deadline)
        if found:
            return found
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _find_decomposition(T, n, m, p, r, deadline):
    # Initialize with random half-integers
    # Use multiple restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.randint(-4, 5, (n*m, r)) * 0.5
        V = np.random.randint(-4, 5, (m*p, r)) * 0.5
        W = np.random.randint(-4, 5, (p*n, r)) * 0.5
        
        # Local search / Gradient descent on the error
        # Error = || T - sum U o V o W ||^2
        # This is a non-convex problem.
        # We can use a simple coordinate descent or Nelder-Mead.
        # Given the small size, Nelder-Mead on flattened array might work.
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*r], x[n*m*r:n*m*r+m*p*r], x[n*m*r+m*p*r:]
            U_curr = U_flat.reshape(n*m, r)
            V_curr = V_flat.reshape(m*p, r)
            W_curr = W_flat.reshape(p*n, r)
            
            # Compute decomposed tensor
            # T_decomp[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            T_decomp = np.zeros((n, m, p))
            for c in range(r):
                u_c = U_curr[:, c]
                v_c = V_curr[:, c]
                w_c = W_curr[:, c]
                # Expand to (n, m, p)
                # u_c is (n*m), v_c is (m*p), w_c is (p*n)
                # We need to map indices carefully.
                for i in range(n):
                    for j in range(m):
                        u_val = u_c[i*m+j]
                        if u_val == 0: continue
                        for k in range(p):
                            v_val = v_c[j*p+k]
                            w_val = w_c[k*n+i]
                            T_decomp[i, j, k] += u_val * v_val * w_val
                            
            diff = T - T_decomp
            return np.sum(diff**2)
            
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use scipy minimize
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
        except Exception:
            continue
            
        if res.fun < 1e-6:
            # Snap to half-integers
            U_snap = np.round(res.x[:n*m*r] * 2) / 2
            V_snap = np.round(res.x[n*m*r:n*m*r+m*p*r] * 2) / 2
            W_snap = np.round(res.x[n*m*r+m*p*r:] * 2) / 2
            
            # Verify exactness
            U_curr = U_snap.reshape(n*m, r)
            V_curr = V_snap.reshape(m*p, r)
            W_curr = W_snap.reshape(p*n, r)
            
            T_decomp = np.zeros((n, m, p))
            for c in range(r):
                u_c = U_curr[:, c]
                v_c = V_curr[:, c]
                w_c = W_curr[:, c]
                for i in range(n):
                    for j in range(m):
                        u_val = u_c[i*m+j]
                        if u_val == 0: continue
                        for k in range(p):
                            v_val = v_c[j*p+k]
                            w_val = w_c[k*n+i]
                            T_decomp[i, j, k] += u_val * v_val * w_val
                            
            if np.allclose(T_decomp, T, atol=1e-9):
                # Convert to lists of fractions/floats as required
                # The interface expects integer/half-integer entries.
                # Returning floats that are exact half-integers is fine.
                return {
                    "U": U_curr.tolist(),
                    "V": V_curr.tolist(),
                    "W": W_curr.tolist()
                }
                
    return None

def _schoolbook(n, m, p):
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
