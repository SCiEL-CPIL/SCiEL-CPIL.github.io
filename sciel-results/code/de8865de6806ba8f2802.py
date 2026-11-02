import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Indices:
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Target Tensor T[u, v, w]
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u, v, w] = 1 if indices are consistent, else 0
    
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                TARGET_TENSOR[u, v, w] = 1.0

    def tensor_contraction(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def verify_solution(U, V, W):
        """Verify that U, V, W correctly decompose the tensor with exact arithmetic."""
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = 0
                    for c in range(len(U[0])):
                        total += U_frac[u_idx][c] * V_frac[v_idx][c] * W_frac[w_idx][c]
                    if total != 1:
                        return False
        return True

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((size_u, rank_school))
    V_school = np.zeros((size_v, rank_school))
    W_school = np.zeros((size_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_solution = None
    best_rank = rank_school

    # Strategy: Try to find a lower rank decomposition using a "splitting" approach.
    # We fix the middle dimension (m=4) and try to find a decomposition that exploits
    # the block structure.
    
    # The tensor can be viewed as a 3x4 matrix of 4x4 blocks.
    # T_{i,j,k} = 1 if (i,j,k) is a valid index triple.
    
    # Let's try a randomized search for a rank < 48 solution.
    # We will use a local optimization approach starting from the schoolbook solution
    # and trying to zero out columns.
    
    # However, a more direct approach for this specific task is to check known results
    # or use a specific algebraic construction if one exists for 3x4*4x4.
    
    # Since 3x4*4x4 is not a standard "winograd" case like 3x3, we rely on numerical search.
    
    # Let's try a simple heuristic:
    # 1. Start with rank 48.
    # 2. Try to reduce rank by 1, 2, ... down to 40.
    # 3. For each rank R, try to find a solution by minimizing the error.
    
    # To make this efficient, we will use a "deflation" approach.
    # We assume that the tensor can be written as a sum of rank-1 terms.
    
    # Let's try a specific known low-rank decomposition if available, or a randomized one.
    # For 3x4 * 4x4, the rank is not well-known to be significantly below 48 in standard literature
    # without specific algebraic tricks.
    
    # Let's try to find a solution with rank 47, 46, etc.
    
    # We will use a simple local search:
    # Start with the schoolbook solution.
    # Pick a random column to "remove" (set to 0).
    # Adjust the other columns to compensate.
    
    # This is a non-linear problem. We will use scipy.optimize.minimize.
    
    from scipy.optimize import minimize
    
    def objective(params, R):
        if time.time() > deadline:
            return 1e10
        
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        
        # Minimize the squared error on the non-zero entries
        # and the zero entries
        return np.sum((T - TARGET_TENSOR)**2)

    # Try ranks from 47 down to 40
    for R in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random parameters
        np.random.seed(42 + R)
        params = np.random.randn(size_u * R + size_v * R + size_w * R) * 0.1
        
        try:
            res = minimize(objective, params, args=(R,), method='L-BFGS-B', 
                           options={'maxiter': 100, 'maxfun': 2000})
            
            if res.fun < 1e-6:
                # Check if it's a valid solution
                U_opt = res.x[:size_u * R].reshape(size_u, R)
                V_opt = res.x[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                W_opt = res.x[size_u * R + size_v * R :].reshape(size_w, R)
                
                # Snap to half-integers?
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify snapped solution
                T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
                    # Verify with exact arithmetic
                    if verify_solution(U_snap, V_snap, W_snap):
                        best_rank = R
                        best_solution = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        break # Found a better solution
        except Exception:
            pass

    if best_solution:
        return best_solution
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
