import numpy as np
import random
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook
    U = np.zeros((n * m, base_rank), dtype=float)
    V = np.zeros((m * p, base_rank), dtype=float)
    W = np.zeros((p * n, base_rank), dtype=float)
    
    col = 0
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                indices.append((i, j, k))
                col += 1
                
    # Try to reduce rank by 1
    target_rank = base_rank - 1
    
    # We will try to find a decomposition of rank target_rank.
    # We initialize with the first target_rank columns of the schoolbook decomposition.
    # This leaves out one term, say the last one (i=2, j=3, k=4).
    # We then optimize the entries of U, V, W to satisfy the constraints.
    
    U0 = U[:, :target_rank].copy()
    V0 = V[:, :target_rank].copy()
    W0 = W[:, :target_rank].copy()
    
    # The missing term is at index (2, 3, 4) in our loop order?
    # Loop order: i=0..2, j=0..3, k=0..4.
    # Last index: i=2, j=3, k=4.
    # u_idx = 2*4 + 3 = 11
    # v_idx = 3*5 + 4 = 19
    # w_idx = 4*3 + 2 = 14
    
    # Define the residual function
    def residual(params):
        # params is a flattened vector of length target_rank * (n*m + m*p + p*n)
        # Split into U, V, W
        u_size = n * m
        v_size = m * p
        w_size = p * n
        
        U_flat = params[:target_rank * u_size]
        V_flat = params[target_rank * u_size : target_rank * (u_size + v_size)]
        W_flat = params[target_rank * (u_size + v_size) :]
        
        U_mat = U_flat.reshape(u_size, target_rank)
        V_mat = V_flat.reshape(v_size, target_rank)
        W_mat = W_flat.reshape(w_size, target_rank)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over r
                    val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                    res[idx] = val - 1
                    idx += 1
        return res

    # Initial guess: the schoolbook columns we kept.
    # We need to flatten them in the same order as expected by residual.
    # The residual function expects U_flat to be row-major? 
    # reshape(u_size, target_rank) implies the input is row-major flattened.
    # U0 is (u_size, target_rank). 
    # If we flatten U0 with 'C' order, it matches.
    
    x0 = np.concatenate([
        U0.flatten(), 
        V0.flatten(), 
        W0.flatten()
    ])
    
    # Run optimization
    # We might need multiple restarts if it gets stuck.
    best_result = None
    best_cost = float('inf')
    
    for _ in range(10):
        # Add small noise to escape local minima
        x0_perturbed = x0 + np.random.normal(0, 0.1, size=x0.shape)
        
        try:
            result = least_squares(residual, x0_perturbed, method='lm') # Levenberg-Marquardt
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            continue
            
    if best_result is not None and best_result.cost < 1e-6:
        # Extract U, V, W
        u_size = n * m
        v_size = m * p
        w_size = p * n
        
        U_sol = best_result.x[:target_rank * u_size].reshape(u_size, target_rank)
        V_sol = best_result.x[target_rank * u_size : target_rank * (u_size + v_size)].reshape(v_size, target_rank)
        W_sol = best_result.x[target_rank * (u_size + v_size) :].reshape(w_size, target_rank)
        
        # Check if we can snap to half-integers
        # The problem requires integer or half-integer entries.
        # Let's try rounding to nearest 0.5
        U_snap = np.round(U_sol * 2) / 2
        V_snap = np.round(V_sol * 2) / 2
        W_snap = np.round(W_sol * 2) / 2
        
        if check_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        # If snapping fails, maybe the solution is not exact half-integers?
        # But for these small dimensions, it likely is.
        # If not, we return the schoolbook.
        
    # Fallback to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
