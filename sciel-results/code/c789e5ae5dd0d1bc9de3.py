import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def verify(U, V, W, r):
        T_comp = np.zeros((n, m, p))
        for c in range(r):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        T_comp[i, j, k] += u_val * v_col[j * p + k] * w_col[k * n + i]
        if not np.allclose(T_comp, 1.0, atol=1e-9):
            return False
        return True

    def snap_to_half_int(X):
        return np.round(X * 2) / 2

    def try_find_decomposition(r, num_tries=200):
        for trial in range(num_tries):
            if time.time() > deadline:
                return None
            
            # Strategy: For each (i,k) pair, we need A_ik * w_ik = 1_m
            # where A_ik[j,c] = U[i*m+j, c] * V[j*p+k, c]
            # We want A_ik to have rank m=3 so that a solution exists.
            
            # Approach: Choose U and V such that for each (i,k), the rows of A_ik are linearly independent.
            # Then w_ik = A_ik^T (A_ik A_ik^T)^{-1} * 1_m
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            W_sol = np.zeros((p * n, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build A_ik (m x r)
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_ik[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # Check if A_ik has full row rank
                    try:
                        # Compute the pseudoinverse solution
                        w_vec = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)[0]
                        # Check residual
                        residual = np.linalg.norm(A_ik @ w_vec - np.ones(m))
                        if residual > 1e-8:
                            valid = False
                            break
                        W_sol[k * n + i, :] = w_vec
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                if verify(U, V, W_sol, r):
                    # Try to snap to half-integers
                    U_snap = snap_to_half_int(U)
                    V_snap = snap_to_half_int(V)
                    W_snap = snap_to_half_int(W_sol)
                    
                    if verify(U_snap, V_snap, W_snap, r):
                        return U_snap, V_snap, W_snap
                    
                    # If snapping fails, try to find a rational approximation
                    # by using the fact that we need exact integer/half-integer entries
                    # For now, if we have a valid real solution, we might need to refine it
                    # But the task requires integer/half-integer, so we may need to fail
                    pass
        
        return None

    # Try to find a decomposition with rank < 45
    # Known results: rank(3x3x5) is at most 45 (schoolbook)
    # Let's try from 44 down
    
    for r in range(44, 35, -1):
        if time.time() > deadline:
            break
        result = try_find_decomposition(r, num_tries=100)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
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
