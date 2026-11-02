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

def _try_rank_deflation(deadline):
    """
    Attempt to find a rank-25 decomposition by deflating one term from the rank-26 solution.
    For each term r to eliminate, fix the other 25 terms and solve for corrections
    that compensate for the removed term.
    """
    n, m, p = DIMENSIONS
    R_orig = 26
    R_target = 25
    T_target = np.ones((n, m, p))
    
    U26, V26, W26 = _get_strassen_26_decomposition()
    
    # For each candidate term to remove, try to find a rank-25 solution
    for remove_idx in range(R_orig):
        if time.time() > deadline:
            break
        
        # Keep all terms except remove_idx
        keep_idx = [i for i in range(R_orig) if i != remove_idx]
        
        # The contribution of the removed term to the tensor
        U_rem = U26[:, remove_idx].reshape(n, m)
        V_rem = V26[:, remove_idx].reshape(m, p)
        W_rem = W26[:, remove_idx].reshape(p, n)
        T_rem = np.einsum('ij,jk,ki->ijk', U_rem, V_rem, W_rem)
        
        # We need the remaining 25 terms to produce T_target + T_rem
        # (since removing term r means the remaining terms must compensate)
        T_needed = T_target + T_rem
        
        # Initialize with the kept terms from the rank-26 solution
        U_init = U26[:, keep_idx].copy()
        V_init = V26[:, keep_idx].copy()
        W_init = W26[:, keep_idx].copy()
        
        # Use least_squares to adjust the 25 terms to match T_needed
        def _loss_25(params):
            U = params[:9*R_target].reshape(9, R_target)
            V = params[9*R_target:18*R_target].reshape(9, R_target)
            W = params[18*R_target:].reshape(9, R_target)
            
            U_reshaped = U.reshape((n, m, R_target))
            V_reshaped = V.reshape((m, p, R_target))
            W_reshaped = W.reshape((p, n, R_target))
            
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            diff = T_recon - T_needed
            return diff.flatten()
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(_loss_25, params_init, method='lm', max_nfev=5000)
            if result.cost < 1e-14:
                # Verify the final decomposition produces exactly T_target
                U_final = result.x[:9*R_target].reshape(9, R_target)
                V_final = result.x[9*R_target:18*R_target].reshape(9, R_target)
                W_final = result.x[18*R_target:].reshape(9, R_target)
                
                if _verify_decomposition(U_final, V_final, W_final, n, m, p, atol=1e-10):
                    return U_final, V_final, W_final
        except Exception:
            pass
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Get the standard Rank-26 solution
    U26, V26, W26 = _get_strassen_26_decomposition()
    
    if not _verify_decomposition(U26, V26, W26, n, m, p):
        # Fallback to standard 27 if verification fails (should not happen)
        U, V, W = _get_standard_27()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 26
    best_result = (U26, V26, W26)
    
    # Attempt rank deflation: remove one term from rank-26 to get rank-25
    result_25 = _try_rank_deflation(deadline)
    if result_25 is not None:
        U25, V25, W25 = result_25
        if _verify_decomposition(U25, V25, W25, n, m, p, atol=1e-10):
            best_rank = 25
            best_result = (U25, V25, W25)
    
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
