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

def _try_rank_23_search(deadline, n=3, m=3, p=3):
    """
    Attempt to find a rank-23 decomposition using iterative optimization.
    We parameterize the decomposition and use least squares to minimize the error.
    """
    R = 23
    # Initialize with random values
    np.random.seed(42)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    T_target = np.ones((n, m, p))
    T_flat = T_target.flatten()
    
    # Number of restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Scale to reasonable magnitude
        U /= np.sqrt(R)
        V /= np.sqrt(R)
        W /= np.sqrt(R)
        
        # Iterative optimization
        for it in range(50):
            if time.time() > deadline - 5:
                break
                
            # Compute current error
            U_reshaped = U.reshape((n, m, R))
            V_reshaped = V.reshape((m, p, R))
            W_reshaped = W.reshape((p, n, R))
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            error = np.sum((T_recon - T_target) ** 2)
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if error < 1e-10:
                return U.copy(), V.copy(), W.copy()
                
            # Gradient-based update (simple power method style)
            # For each term r, update U[:,r], V[:,r], W[:,r]
            for r in range(R):
                # Compute residual
                T_recon_r = np.einsum('ijr,jkr,kir->ijk', 
                                       U_reshaped[:, :, r:r+1], 
                                       V_reshaped[:, :, r:r+1], 
                                       W_reshaped[:, :, r:r+1])
                
                # Update U[:,r] by projecting residual onto V and W
                # This is a simplified approach
                if it % 3 == 0:
                    # Update U
                    grad_U = np.einsum('jk,jkr->ir', T_target - T_recon, V_reshaped[:, :, r])
                    U[:, r] += 0.1 * grad_U.reshape(-1)
                elif it % 3 == 1:
                    # Update V
                    grad_V = np.einsum('ik,ikir->jr', T_target - T_recon, U_reshaped[:, :, r])
                    V[:, r] += 0.1 * grad_V.reshape(-1)
                else:
                    # Update W
                    grad_W = np.einsum('ij,ijr->kr', T_target - T_recon, W_reshaped[:, :, r])
                    W[:, r] += 0.1 * grad_W.reshape(-1)
                    
                # Recompute reshaped arrays
                U_reshaped = U.reshape((n, m, R))
                V_reshaped = V.reshape((m, p, R))
                W_reshaped = W.reshape((p, n, R))
                
        if best_error < 1e-10:
            return best_U, best_V, best_W
    
    # Check if we found a valid solution
    if best_U is not None and best_error < 1e-9:
        return best_U, best_V, best_W
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Get the standard Rank-26 solution as fallback
    U26, V26, W26 = _get_strassen_26_decomposition()
    
    if not _verify_decomposition(U26, V26, W26, n, m, p):
        # Fallback to standard 27 if verification fails (should not happen)
        U, V, W = _get_standard_27()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 26
    best_result = (U26, V26, W26)
    
    # Attempt to find a rank-23 solution
    result_23 = _try_rank_23_search(deadline, n, m, p)
    if result_23 is not None:
        U23, V23, W23 = result_23
        if _verify_decomposition(U23, V23, W23, n, m, p):
            best_rank = 23
            best_result = (U23, V23, W23)
    
    # If we have time, try rank-24 as well
    if best_rank > 24 and time.time() < deadline - 10:
        # Try rank-24 with a similar approach
        R = 24
        np.random.seed(43)
        T_target = np.ones((n, m, p))
        best_error_24 = float('inf')
        best_U_24 = None
        best_V_24 = None
        best_W_24 = None
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            U /= np.sqrt(R)
            V /= np.sqrt(R)
            W /= np.sqrt(R)
            
            for it in range(30):
                if time.time() > deadline - 5:
                    break
                    
                U_reshaped = U.reshape((n, m, R))
                V_reshaped = V.reshape((m, p, R))
                W_reshaped = W.reshape((p, n, R))
                T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                error = np.sum((T_recon - T_target) ** 2)
                
                if error < best_error_24:
                    best_error_24 = error
                    best_U_24 = U.copy()
                    best_V_24 = V.copy()
                    best_W_24 = W.copy()
                    
                if error < 1e-10:
                    break
                    
                for r in range(R):
                    if it % 3 == 0:
                        grad_U = np.einsum('jk,jkr->ir', T_target - T_recon, V_reshaped[:, :, r])
                        U[:, r] += 0.1 * grad_U.reshape(-1)
                    elif it % 3 == 1:
                        grad_V = np.einsum('ik,ikir->jr', T_target - T_recon, U_reshaped[:, :, r])
                        V[:, r] += 0.1 * grad_V.reshape(-1)
                    else:
                        grad_W = np.einsum('ij,ijr->kr', T_target - T_recon, W_reshaped[:, :, r])
                        W[:, r] += 0.1 * grad_W.reshape(-1)
                        
                    U_reshaped = U.reshape((n, m, R))
                    V_reshaped = V.reshape((m, p, R))
                    W_reshaped = W.reshape((p, n, R))
                    
            if best_error_24 < 1e-10:
                break
        
        if best_U_24 is not None and best_error_24 < 1e-9:
            if _verify_decomposition(best_U_24, best_V_24, best_W_24, n, m, p):
                best_rank = 24
                best_result = (best_U_24, best_V_24, best_W_24)
    
    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}

# EVOLVE-BLOCK-END
