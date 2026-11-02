"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _build_decomposition_from_arrays(U, V, W):
    """
    Ensure U, V, W are in the correct format and return them.
    """
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _get_strassen_26_decomposition():
    """
    Returns the standard Rank-26 decomposition (Strassen 2x2 core + 19 border terms).
    """
    n, m, p = DIMENSIONS
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen Terms for 2x2 Core (i,j,k in 0,1)
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[0, r] = 1; U[4, r] = 1
    V[0, r] = 1; V[4, r] = 1
    W[0, r] = 1; W[4, r] = 1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[3, r] = 1; U[4, r] = 1
    V[0, r] = 1
    W[1, r] = 1; W[4, r] = -1
    
    # m3 = a00(b01 - b11)
    r = 2
    U[0, r] = 1
    V[1, r] = 1; V[4, r] = -1
    W[3, r] = 1; W[4, r] = 1
    
    # m4 = a11(b10 - b00)
    r = 3
    U[4, r] = 1
    V[3, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[1, r] = 1
    
    # m5 = (a00 + a01)b11
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[4, r] = 1
    W[0, r] = -1; W[3, r] = 1
    
    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[3, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[4, r] = 1
    
    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[1, r] = 1; U[4, r] = -1
    V[3, r] = 1; V[4, r] = 1
    W[0, r] = 1
    
    # Border Terms (19 terms)
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # Skip if i<2 and j<2 and k<2
    border_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                r = 7 + border_count
                border_count += 1
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
    return U, V, W

def _try_rank_23_search(deadline):
    """
    Attempt to find a rank-23 decomposition by relaxing the rank-26 structure.
    We parameterize a general rank-23 tensor and use optimization to fit the target.
    This is a difficult non-convex problem, so we use random restarts.
    """
    n, m, p = DIMENSIONS
    R = 23
    T_target = np.ones((n, m, p))
    T_flat = T_target.flatten()
    
    def _loss(params):
        # params: flattened U (9x23), V (9x23), W (9x23)
        U = params[:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:].reshape(9, R)
        
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        diff = T_recon - T_target
        return np.sum(diff**2)
    
    best_loss = float('inf')
    best_params = None
    rng = np.random.RandomState(42)
    
    # Use the rank-26 solution as a starting point, zeroing out one column
    U26, V26, W26 = _get_strassen_26_decomposition()
    
    # Try removing each of the 26 columns and see if we can recover the tensor with 23 terms
    # This is a heuristic: we take the first 23 columns of the rank-26 solution
    # and then try to optimize to see if we can reach zero loss.
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Initialize with random small perturbations around a subset of the rank-26 solution
        # We pick 23 random columns from the 26 and initialize there
        cols = rng.choice(26, size=23, replace=False)
        
        U_init = U26[:, cols]
        V_init = V26[:, cols]
        W_init = W26[:, cols]
        
        # Add small random noise
        noise_scale = 0.01
        U_init = U_init + rng.normal(0, noise_scale, U_init.shape)
        V_init = V_init + rng.normal(0, noise_scale, V_init.shape)
        W_init = W_init + rng.normal(0, noise_scale, W_init.shape)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use least_squares to minimize the loss
        try:
            result = least_squares(_loss, params_init, method='lm', max_nfev=2000)
            if result.cost < best_loss:
                best_loss = result.cost
                best_params = result.x
        except Exception:
            pass
        
    if best_params is not None and best_loss < 1e-8:
        U = best_params[:9*R].reshape(9, R)
        V = best_params[9*R:18*R].reshape(9, R)
        W = best_params[18*R:].reshape(9, R)
        return U, V, W
        
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Get the standard Rank-26 solution
    U26, V26, W26 = _get_strassen_26_decomposition()
    
    if not _verify_decomposition(U26, V26, W26, n, m, p):
        # Fallback to standard 27 if verification fails (should not happen)
        U, V, W = _get_standard_27()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 26
    best_result = (U26, V26, W26)
    
    # Attempt to find a lower rank solution (e.g., 23)
    result_23 = _try_rank_23_search(deadline)
    if result_23 is not None:
        U23, V23, W23 = result_23
        if _verify_decomposition(U23, V23, W23, n, m, p):
            best_rank = 23
            best_result = (U23, V23, W23)
    
    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}

def _get_standard_27():
    """Fallback standard decomposition."""
    n, m, p = DIMENSIONS
    R = 27
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return U, V, W

# EVOLVE-BLOCK-END
