"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook decomposition for 2x3x4 matrix multiplication.
    Attempts to find a lower rank decomposition if possible within time limits.
    Rank = 24 (schoolbook) or lower if found.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Generate the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition
    def get_schoolbook():
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_solution = get_schoolbook()
    best_rank = n * m * p
    
    # Time guard
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Attempt to find rank 23
    # We use a local search approach: initialize with random small perturbations of schoolbook
    # and try to drive error to 0 with one fewer term.
    # This is heuristic and may fail, but safe if it falls back.
    
    target_rank = best_rank - 1
    if target_rank < 1:
        target_rank = 1
        
    # We will try to solve for rank 23
    # Variables: flattened U, V, W of size (nm + mp + pn) * target_rank
    # Objective: sum( (T - tensor_product(U,V,W))**2 )
    
    def tensor_product(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Result: (nm, mp, pn)
        # Reshape and einsum
        r = U.shape[1]
        # U_i,t * V_j,t * W_k,t
        # np.einsum('it,jt,kt->ijk', U, V, W)
        return np.einsum('it,jt,kt->ijk', U, V, W)

    def error_func(params):
        if time.time() - start_time > timeout:
            return float('inf')
        
        r = target_rank
        size_u = nm * r
        size_v = mp * r
        size_w = pn * r
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:size_u + size_v + size_w]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        
        T_calc = tensor_product(U, V, W)
        diff = T - T_calc
        return np.sum(diff * diff)

    # Initialization: Take schoolbook, drop one column, and scale/adjust?
    # Better: Random initialization or schoolbook with noise.
    # Schoolbook has 24 columns. We need 23.
    # We can initialize the 23 columns with the first 23 columns of schoolbook and add noise.
    
    U_sb = best_solution['U']
    V_sb = best_solution['V']
    W_sb = best_solution['W']
    
    # Drop the last column
    U_init = U_sb[:, :target_rank].copy()
    V_init = V_sb[:, :target_rank].copy()
    W_init = W_sb[:, :target_rank].copy()
    
    # Add small noise to escape local minima
    U_init += np.random.normal(0, 0.1, U_init.shape)
    V_init += np.random.normal(0, 0.1, V_init.shape)
    W_init += np.random.normal(0, 0.1, W_init.shape)
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Use L-BFGS-B
    try:
        res = minimize(error_func, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 2000})
        
        if res.success or (res.fun < 1e-6):
            # Check if we can snap to half-integers
            r = target_rank
            size_u = nm * r
            size_v = mp * r
            size_w = pn * r
            
            U_opt = res.x[:size_u].reshape(nm, r)
            V_opt = res.x[size_u:size_u + size_v].reshape(mp, r)
            W_opt = res.x[size_u + size_v:].reshape(pn, r)
            
            # Snap to nearest half-integer
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            # Verify exactness
            T_calc = tensor_product(U_snap, V_snap, W_snap)
            if np.allclose(T_calc, T, atol=1e-9):
                # Valid solution found
                best_solution = {"U": U_snap, "V": V_snap, "W": W_snap}
                best_rank = r
    except Exception:
        pass

    return best_solution
# EVOLVE-BLOCK-END
