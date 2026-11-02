import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Indices:
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i
    
    size_u = n * m  # 12
    size_v = m * p  # 16
    size_w = p * n  # 12
    
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

    # Strategy: Use a randomized algebraic search with Gaussian elimination
    # to find linear dependencies in the tensor space.
    
    # The key insight: if we can find a set of rank-1 tensors that span the
    # multiplication tensor, we have a decomposition. We search for such a set
    # by generating random rank-1 candidates and checking for linear dependencies.
    
    # We'll use a different approach: try to find a decomposition with rank < 48
    # by solving a system of equations.
    
    # For each rank R from 47 down to 40, we try to find U, V, W such that
    # sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w] for all u,v,w.
    
    # This is a system of size_u * size_v * size_w = 12*16*12 = 2304 equations
    # with 3*R variables (for each r: U has 12, V has 16, W has 12 variables).
    
    # We'll use a nonlinear solver, but initialize with a structured guess.
    
    from scipy.optimize import minimize
    
    def objective(params, R):
        if time.time() > deadline:
            return 1e10
        
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        
        # Weighted error: emphasize the non-zero entries
        mask = TARGET_TENSOR != 0
        error = np.sum((T[mask] - TARGET_TENSOR[mask])**2) + 0.1 * np.sum((T[~mask])**2)
        return error

    def gradient(params, R):
        if time.time() > deadline:
            return np.zeros_like(params)
        
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        error = T - TARGET_TENSOR
        
        # Gradient with respect to U
        grad_U = np.einsum('uvw,vr,wr->ur', error, V, W)
        grad_V = np.einsum('uvw,ur,wr->vr', error, U, W)
        grad_W = np.einsum('uvw,ur,vr->wr', error, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Try ranks from 47 down to 42
    for R in range(47, 41, -1):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + R * 10 + restart)
            
            # Initialize with a structured pattern
            params = np.random.randn(size_u * R + size_v * R + size_w * R) * 0.5
            
            try:
                res = minimize(objective, params, args=(R,), method='L-BFGS-B', 
                               jac=gradient,
                               options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:size_u * R].reshape(size_u, R)
                    V_opt = res.x[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                    W_opt = res.x[size_u * R + size_v * R :].reshape(size_w, R)
                    
                    # Snap to half-integers
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
                            break  # Found a better solution
            except Exception:
                pass
        
        if best_solution and best_rank < rank_school:
            break  # Found a solution with rank < 48

    if best_solution:
        return best_solution
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
